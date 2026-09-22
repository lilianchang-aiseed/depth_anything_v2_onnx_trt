#!/usr/bin/env python3
"""Capture DPT refinenet4..1 outputs without changing the model forward pass.

The montage visualizes each decoder activation after multiplication by
``dmax``.  It uses the channel-wise mean absolute activation, which is useful
for locating spatial responses but is not itself a depth prediction.  Decoder
activations and the final prediction are not saved as NPY files.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


METRIC_DIR = Path(__file__).resolve().parent
if str(METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC_DIR))

from model_metric_depth.dpt import DepthAnythingV2 as SoftplusDepthAnythingV2
from model_metric_depth.dpt_custom import DepthAnythingV2 as CustomDepthAnythingV2
from model_metric_depth.dpt_vkitti import VKITTIDepthAnythingV2
from model_metric_depth.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from torchvision.transforms import Compose


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,
             'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128,
             'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256,
             'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384,
             'out_channels': [1536, 1536, 1536, 1536]},
}


def _strip_module(state_dict):
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in state_dict.items()}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError('checkpoint must contain a model state_dict')
    state_dict = _strip_module(checkpoint['model'])
    encoder = checkpoint.get('encoder', 'vits')
    max_depth = float(checkpoint.get('max_depth', 20.0))
    model_config = checkpoint.get('model_config') or {}
    architecture_id = model_config.get('architecture_id')

    if architecture_id == 'fisheye_metric_dpt_custom_softplus_v1':
        model_class = CustomDepthAnythingV2
        selected = 'dpt-custom'
    elif (architecture_id == 'dav2_metric_vkitti_sigmoid_v1' or
          (not architecture_id and
           'depth_head.scratch.output_conv2.0.weight' in state_dict)):
        model_class = VKITTIDepthAnythingV2
        selected = 'metric-sigmoid'
    else:
        model_class = SoftplusDepthAnythingV2
        selected = 'dpt-softplus'

    model = model_class(
        **MODEL_CONFIGS[encoder], max_depth=max_depth).to(device).eval()
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint, selected


def read_bgr(path):
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as sample:
            if 'left' not in sample.files:
                raise KeyError(f'{path} has no left array')
            image = np.asarray(sample['left'])
            if image.ndim == 2:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            return np.ascontiguousarray(image[..., :3]).astype(np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot read image: {path}')
    return image


def preprocess(bgr, size):
    transform = Compose([
        Resize(width=size, height=size, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound',
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    array = transform({'image': rgb})['image']
    return torch.from_numpy(array).unsqueeze(0)


def activation_heatmap(activation, out_hw, dmax):
    feature = np.mean(np.abs(activation[0]), axis=0)
    # Use one fixed absolute scale. Per-image percentile normalization would
    # cancel multiplication by dmax and make both visualizations identical.
    norm = np.clip(feature / max(float(dmax), 1e-12), 0.0, 1.0)
    color = cv2.applyColorMap(
        np.round(norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.resize(color, (out_hw[1], out_hw[0]),
                       interpolation=cv2.INTER_LINEAR)
    return color, float(feature.min()), float(feature.max())


def depth_heatmap(depth, out_hw, dmin, dmax):
    norm = np.clip((depth - dmin) / max(dmax - dmin, 1e-12), 0.0, 1.0)
    color = cv2.applyColorMap(
        np.round(norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.resize(color, (out_hw[1], out_hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def label(panel, text):
    cv2.putText(panel, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def inspect_one(input_path, output_path, model, selected, device, size,
                dmin, dmax):
    bgr = read_bgr(input_path)
    tensor = preprocess(bgr, size).to(device)

    activations = {}
    handles = []
    for number in (4, 3, 2, 1):
        module = getattr(model.depth_head.scratch, f'refinenet{number}')

        def capture(_module, _inputs, output, key=f'path_{number}'):
            activations[key] = output.detach().float().cpu().numpy()

        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            prediction = model(tensor)[0].detach().float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()

    scaled_activations = {}
    for key in ('path_4', 'path_3', 'path_2', 'path_1'):
        value = activations[key]
        scaled_activations[key] = value #* dmax

    out_hw = (size, size)
    input_panel = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    panels = [label(input_panel, f'Input: {input_path.name}')]
    for key in ('path_4', 'path_3', 'path_2', 'path_1'):
        shape = activations[key].shape
        panel, value_min, value_max = activation_heatmap(
            scaled_activations[key], out_hw, dmax)
        panels.append(label(
            panel,
            f'{key} *{dmax:g} [{value_min:.2f}, {value_max:.2f}] '
            f'{shape[-2]}x{shape[-1]}'))
    panels.append(label(depth_heatmap(
        prediction, out_hw, dmin, dmax), 'Prediction metric depth'))
    if not cv2.imwrite(str(output_path), np.hstack(panels)):
        raise RuntimeError(f'Failed to write visualization: {output_path}')
    print(f'[{input_path.name}] model={selected} device={device} '
          f'input={tuple(tensor.shape)} prediction={prediction.shape} '
          f'range=[{prediction.min():.4f}, {prediction.max():.4f}] m')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DPT decoder-stage outputs')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path,
                        help='input image, NPZ containing data["left"], or '
                             'folder of NPZ files')
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--img-size', type=int, default=None,
                        help='defaults to checkpoint img_size')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                        default='auto')
    parser.add_argument('--dmin', type=float, default=0.2)
    parser.add_argument('--dmax', type=float, default=20.0)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested, but CUDA is unavailable')

    model, checkpoint, selected = load_model(args.checkpoint, device)
    size = int(args.img_size or checkpoint.get('img_size', 518))
    if size % 14:
        raise SystemExit(f'img-size must be divisible by 14, got {size}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.input.is_dir():
        input_paths = sorted(args.input.glob('*.npz'))
        if not input_paths:
            raise SystemExit(f'No NPZ files found in {args.input}')
        for index, input_path in enumerate(input_paths, 1):
            output_path = (
                args.out_dir / f'{input_path.stem}_decoder_montage.png')
            if output_path.exists():
                print(f'[{index}/{len(input_paths)}] skip existing '
                      f'{output_path.name}')
                continue
            print(f'[{index}/{len(input_paths)}] {input_path}')
            inspect_one(input_path, output_path, model, selected, device, size,
                        args.dmin, args.dmax)
    else:
        output_path = args.out_dir / f'{args.input.stem}_decoder_montage.png'
        if output_path.exists():
            print(f'skip existing {output_path}')
        else:
            inspect_one(args.input, output_path, model, selected, device, size,
                        args.dmin, args.dmax)
    print(f'outputs: {args.out_dir}')


if __name__ == '__main__':
    main()
