"""Train, validate, and finally test a fisheye metric-depth model.

Validation selects ``best.pt``. The held-out test split is never used for
optimization or checkpoint selection; it is evaluated once with ``best.pt``
after training finishes.

Example:
    python train_fisheye_with_test.py \
        --data /path/to/npz_dataset \
        --encoder vitb --img-size 518 \
        --min-depth 0.2 --metric-depth-max 15 \
        --far-depth 19 --max-depth 20 \
        --bs 4 --epochs 60 --lr 5e-6 --amp \
        --pretrained-from ../checkpoints/depth_anything_v2_vitb.pth \
        --save-path runs/fisheye_vitb

Explicit training splits require all three of ``--train-list``, ``--val-list``,
and ``--test-list``. Inference-only mode needs only ``--test-list``::

    python train_fisheye_with_test.py --model-training false \
        --test-list split/test.txt --checkpoint runs/example/best.pt \
        --save-path runs/example_test
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.fisheye_npz import FisheyeNPZ
from model_metric_depth.dpt import DepthAnythingV2 as DefaultDepthAnythingV2
from util.loss import RangeWeightedSiLogLoss, SiLogLoss
from util.metric import eval_depth, eval_mae_ranges

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

MODEL_ARCHITECTURES = {
    'dpt': {
        'architecture_id': 'fisheye_metric_softplus_v1',
        'module': 'model_metric_depth.dpt',
    },
    'dpt-custom': {
        'architecture_id': 'fisheye_metric_dpt_custom_softplus_v1',
        'module': 'model_metric_depth.dpt_custom',
    },
}


def get_model_class(name):
    """Resolve model variants explicitly; experimental modules stay optional."""
    if name == 'dpt':
        return DefaultDepthAnythingV2
    if name == 'dpt-custom':
        from model_metric_depth.dpt_custom import DepthAnythingV2
        return DepthAnythingV2
    raise ValueError(f'unknown model architecture: {name}')


def build_model(args):
    model_class = get_model_class(args.model_architecture)
    return model_class(**{**MODEL_CONFIGS[args.encoder],
                          'max_depth': args.max_depth})

OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_WEIGHT_DECAY = 0.001
HEAD_LR_MULTIPLIER = 10.0
LR_SCHEDULE_POWER = 0.9
SILOG_LAMBDA = 0.5

METRIC_KEYS = [
    'd1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog',
    'mae', 'mae_0.2_2m', 'mae_2_5m', 'mae_5_10m', 'mae_10_20m',
    'silog_0.2_2m', 'silog_2_5m', 'silog_5_10m', 'silog_10_20m',
    'median_relative_error', 'valid_pixel_count',
]

# Exact metric GT is used through 15 m. The last coarse range deliberately
# evaluates the complete stored labels, including far=19 m and sky=20 m.
TEST_COARSE_RANGES = (
    ('0.2_2m', 0.2, 2.0),
    ('2_5m', 2.0, 5.0),
    ('5_10m', 5.0, 10.0),
    ('10_15m', 10.0, 15.0),
    ('15_20m', 15.0, 20.000001),
)
TEST_PER_METER_RANGES = (
    [('0.2_1m', 0.2, 1.0)]
    + [(f'{metre}_{metre + 1}m', float(metre), float(metre + 1))
       for metre in range(1, 19)]
    + [('19_20m', 19.0, 20.000001)]
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def load_training_config(path):
    """Load argparse defaults from either an input or saved run YAML.

    Saved run configs contain resolved/output metadata in addition to the flat
    parser inputs.  Normalize the few historical representations here so an
    old run's ``config.yaml`` can be passed back through ``--config`` without
    weakening the unknown-key check for genuine input mistakes.
    """
    if not path:
        return {}, None
    try:
        import yaml
    except ImportError as error:
        raise SystemExit('PyYAML is required when --config is used') from error
    config_path = os.path.abspath(os.path.expanduser(path))
    with open(config_path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise SystemExit(f'training config must be a YAML mapping: {config_path}')

    # These keys describe what a completed run resolved or constructed; they
    # are not argparse inputs and must not become parser defaults.
    output_only_keys = {
        'model_structure', 'optimizer', 'resolved_split',
        'resolved_split_counts', 'loss_details', 'model_description',
    }

    # Newer saved configs contain both the flat ``model_architecture`` input
    # and detailed ``model_structure`` metadata.  Some intermediate versions
    # only wrote the latter, so recover its parser value before discarding it.
    model_structure = config.get('model_structure')
    if ('model_architecture' not in config
            and isinstance(model_structure, dict)
            and model_structure.get('selection')):
        config['model_architecture'] = model_structure['selection']

    # Historical run configs replaced the parser's string ``loss`` value with
    # a descriptive mapping.  Convert it back to the accepted CLI spelling.
    loss = config.get('loss')
    if isinstance(loss, dict):
        loss_name = str(loss.get('name', '')).strip().lower()
        legacy_loss_names = {
            'silogloss': 'silog',
            'silog': 'silog',
            'rangeweightedsilogloss': 'range-weighted-silog',
            'range-weighted-silog': 'range-weighted-silog',
        }
        if loss_name not in legacy_loss_names:
            raise SystemExit(
                f'{config_path}: unsupported saved loss metadata name '
                f'{loss.get("name")!r}')
        config['loss'] = legacy_loss_names[loss_name]

    for key in output_only_keys:
        config.pop(key, None)

    config_dir = os.path.dirname(config_path)
    path_keys = {
        'data', 'train_list', 'val_list', 'test_list', 'pretrained_from',
        'resume', 'checkpoint', 'save_path', 'vkitti_checkpoint',
    }
    for key in path_keys & config.keys():
        value = config[key]
        if value and not os.path.isabs(os.path.expanduser(str(value))):
            config[key] = os.path.abspath(os.path.join(config_dir, str(value)))
    return config, config_path


# --------------------------------------------------------------------------- #
# Data splitting and experiment records
# --------------------------------------------------------------------------- #
def natural_path_key(path):
    """Sort numbered filenames as 1, 2, ..., 10 instead of 1, 10, ..., 2."""
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r'(\d+)', os.path.abspath(path)))


def contiguous_block_split(files, blocks=10, seed=0):
    """Split sorted frames into blocks, then select whole val/test blocks.

    With ten blocks this is an approximately 8:1:1 split. Validation and test
    each remain one continuous sequence instead of containing random frames.
    """
    files = sorted((os.path.abspath(path) for path in files), key=natural_path_key)
    if len(files) < 3:
        raise ValueError('at least 3 NPZ files are required for train/val/test')
    if blocks < 3:
        raise ValueError('--split-blocks must be at least 3')

    block_count = min(int(blocks), len(files))
    quotient, remainder = divmod(len(files), block_count)
    groups, offset = [], 0
    for index in range(block_count):
        size = quotient + (1 if index < remainder else 0)
        groups.append(files[offset:offset + size])
        offset += size

    # Prefer non-adjacent validation and test blocks so they are not contiguous
    # with one another. The seed makes the choice reproducible.
    choices = [(val_index, test_index)
               for val_index in range(block_count)
               for test_index in range(block_count)
               if val_index != test_index and abs(val_index - test_index) > 1]
    if not choices:
        choices = [(val_index, test_index)
                   for val_index in range(block_count)
                   for test_index in range(block_count)
                   if val_index != test_index]
    val_index, test_index = random.Random(seed).choice(choices)
    train = [path for index, group in enumerate(groups)
             if index not in (val_index, test_index) for path in group]
    return train, groups[val_index], groups[test_index], {
        'method': 'contiguous_blocks',
        'block_count': block_count,
        'validation_block': val_index,
        'test_block': test_index,
        'seed': seed,
    }


def read_list(path):
    with open(path, 'r') as f:
        return [os.path.abspath(line.strip()) for line in f if line.strip()]


def write_list(paths, out_path):
    with open(out_path, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    return out_path


def save_run_metadata(args, counts, split_details):
    config = dict(vars(args))
    architecture = MODEL_ARCHITECTURES[args.model_architecture]
    config['model_structure'] = {
        'selection': args.model_architecture,
        'architecture_id': architecture['architecture_id'],
        'model_class': 'DepthAnythingV2',
        'model_module': architecture['module'],
        'head_type': 'custom_softplus',
        'output_activation': 'softplus',
        'bounded_output': False,
    }
    config['resolved_split_counts'] = counts
    config['resolved_split'] = split_details
    config['optimizer'] = {
        'name': 'AdamW',
        'encoder_lr': args.lr,
        'head_lr': args.lr * HEAD_LR_MULTIPLIER,
        'head_lr_multiplier': HEAD_LR_MULTIPLIER,
        'betas': list(OPTIMIZER_BETAS),
        'weight_decay': OPTIMIZER_WEIGHT_DECAY,
        'lr_schedule': 'polynomial_decay',
        'lr_schedule_power': LR_SCHEDULE_POWER,
    }
    # Keep ``loss`` as the flat parser value so this resolved config remains a
    # valid future --config input.  Store the expanded record separately.
    config['loss_details'] = {
        'name': ('SiLogLoss' if args.loss == 'silog'
                 else 'RangeWeightedSiLogLoss'),
        'lambda': SILOG_LAMBDA,
        'range_weights': {
            '0.2_2m': args.silog_0_2_weight,
            '2_5m': args.silog_2_5_weight,
            '5_10m': args.silog_5_10_weight,
            '10_20m': args.silog_10_20_weight,
        },
        'far_name': 'one-sided ReLU(far_depth - prediction)',
        'far_weight': args.far_loss_weight,
    }
    # JSON is valid YAML and avoids introducing PyYAML as a dependency.
    with open(os.path.join(args.save_path, 'config.yaml'), 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = 'unknown'
    with open(os.path.join(args.save_path, 'git_commit.txt'), 'w') as f:
        f.write(commit + '\n')


def append_csv(path, row):
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline='') as f:
            old_fields = next(csv.reader(f))
        if old_fields != list(row):
            raise RuntimeError(f'CSV columns differ from existing file: {path}')
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# Checkpoints / pretrained weights
# --------------------------------------------------------------------------- #
def _strip_module(state_dict):
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in state_dict.items()}


def load_pretrained(model, path, load_head):
    raw = torch.load(path, map_location='cpu')
    state_dict = raw.get('model', raw) if isinstance(raw, dict) else raw
    state_dict = _strip_module(state_dict)
    if not load_head:
        state_dict = {key: value for key, value in state_dict.items()
                      if 'pretrained' in key}
    result = model.load_state_dict(state_dict, strict=False)
    print(f'[pretrained] {os.path.basename(path)} load_head={load_head}: '
          f'loaded={len(state_dict)} missing={len(result.missing_keys)} '
          f'unexpected={len(result.unexpected_keys)}')


def checkpoint_metadata(model, epoch, best, args):
    metadata = {
        'model': model.state_dict(),
        'epoch': epoch,
        'best': best,
        'encoder': args.encoder,
        'img_size': args.img_size,
        'min_depth': args.min_depth,
        'metric_depth_max': args.metric_depth_max,
        'far_depth': args.far_depth,
        'max_depth': args.max_depth,
        'target_key': args.target_key,
        'input_mode': args.input_mode,
        'loss': args.loss,
        'silog_range_weights': [
            args.silog_0_2_weight, args.silog_2_5_weight,
            args.silog_5_10_weight, args.silog_10_20_weight,
        ],
    }
    if hasattr(model, 'get_model_config'):
        metadata['model_config'] = model.get_model_config()
    return metadata


def save_best(path, model, epoch, best, args):
    """Save a deployment-oriented checkpoint without optimizer state."""
    torch.save(checkpoint_metadata(model, epoch, best, args), path)


def save_last(path, model, optimizer, scaler, epoch, best, best_epoch, args):
    """Save every state needed to resume training."""
    state = checkpoint_metadata(model, epoch, best, args)
    state['best_epoch'] = best_epoch
    state['optimizer'] = optimizer.state_dict()
    state['scaler'] = scaler.state_dict()
    torch.save(state, path)


def validate_checkpoint_architecture(checkpoint, model, label='checkpoint'):
    """Reject known architecture mismatches while accepting legacy files."""
    if not isinstance(checkpoint, dict):
        return
    saved = checkpoint.get('model_config')
    if not saved or not hasattr(model, 'get_model_config'):
        return
    expected = model.get_model_config()
    saved_id = saved.get('architecture_id')
    expected_id = expected.get('architecture_id')
    if saved_id and expected_id and saved_id != expected_id:
        raise SystemExit(
            f'{label} architecture={saved_id!r}, but the selected model uses '
            f'{expected_id!r}. Select the matching model class.')


# --------------------------------------------------------------------------- #
# Visualization and evaluation
# --------------------------------------------------------------------------- #
def _colorize(depth, mask, dmin, dmax):
    depth = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    mask = (mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask)
            else np.asarray(mask, bool))
    mask &= np.isfinite(depth)
    norm = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~mask] = 40
    return color


def _display_input(image, out_hw):
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    rgb = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb.transpose(1, 2, 0)[..., ::-1])
    return cv2.resize(bgr, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)


def _depth_colorbar(height, dmin, dmax):
    bar_w, label_w = 18, 55
    values = np.linspace(dmax, dmin, height, dtype=np.float32)[:, None]
    norm = np.clip((values - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    canvas = np.full((height, bar_w + label_w, 3), 40, dtype=np.uint8)
    canvas[:, :bar_w] = np.repeat(colors, bar_w, axis=1)
    for y, value in [(14, dmax), (height // 2, (dmin + dmax) / 2),
                     (height - 5, dmin)]:
        cv2.putText(canvas, f'{value:g}m', (bar_w + 3, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return canvas


def save_vis(path, image, depth_gt, valid, pred, dmin, dmax, display_gt=None):
    # Metrics/loss use ``valid``; visualization shows the complete GT stored in
    # the NPZ, including metric, far, and sky labels. Zero/NaN stays gray.
    display_gt = depth_gt if display_gt is None else display_gt
    display_valid = torch.isfinite(display_gt) & (display_gt > 0)
    gt_color = _colorize(display_gt, display_valid, dmin, dmax)
    pred_color = _colorize(pred, torch.ones_like(pred, dtype=torch.bool), dmin, dmax)
    input_color = _display_input(image, gt_color.shape[:2])
    for panel, text in [(input_color, 'Input image'),
                        (gt_color, 'GT metric depth'),
                        (pred_color, 'Prediction')]:
        cv2.putText(panel, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    colorbar = _depth_colorbar(gt_color.shape[0], dmin, dmax)
    cv2.imwrite(path, np.hstack([input_color, gt_color, pred_color, colorbar]))


def save_comparison_vis(path, image, depth_gt, custom_pred, vkitti_pred,
                        dmin, dmax, display_gt=None):
    """Input | full GT | custom prediction | official VKITTI prediction."""
    display_gt = depth_gt if display_gt is None else display_gt
    display_valid = torch.isfinite(display_gt) & (display_gt > 0)
    gt_color = _colorize(display_gt, display_valid, dmin, dmax)
    all_custom = torch.ones_like(custom_pred, dtype=torch.bool)
    all_vkitti = torch.ones_like(vkitti_pred, dtype=torch.bool)
    custom_color = _colorize(custom_pred, all_custom, dmin, dmax)
    vkitti_color = _colorize(vkitti_pred, all_vkitti, dmin, dmax)
    input_color = _display_input(image, gt_color.shape[:2])
    panels = [
        (input_color, 'Input image'),
        (gt_color, 'GT metric depth'),
        (custom_color, 'Custom prediction'),
        (vkitti_color, 'Official VKITTI prediction'),
    ]
    for panel, label in panels:
        cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    colorbar = _depth_colorbar(gt_color.shape[0], dmin, dmax)
    cv2.imwrite(path, np.hstack([panel for panel, _ in panels] + [colorbar]))


def _new_eval_accumulator():
    return {
        'totals': {key: 0.0 for key in METRIC_KEYS},
        'counts': {key: 0 for key in METRIC_KEYS},
        'frames': 0,
        'far_pixels': 0,
        'far_violations': 0,
        'far_prediction_sum': 0.0,
    }


def _accumulate_prediction(acc, pred, depth, valid, far, args):
    finite_far = (far == 1) & torch.isfinite(pred)
    if finite_far.any():
        far_values = pred[finite_far]
        count = int(finite_far.sum())
        acc['far_pixels'] += count
        acc['far_violations'] += int((far_values < args.far_depth).sum())
        acc['far_prediction_sum'] += float(far_values.sum())
    mask = ((valid == 1) & (depth >= args.min_depth) &
            (depth <= args.metric_depth_max) & torch.isfinite(pred))
    if mask.sum() < 10:
        return mask, False
    result = eval_depth(pred[mask], depth[mask], extended=True)
    for key in METRIC_KEYS:
        value = result[key]
        if key == 'valid_pixel_count':
            acc['totals'][key] += value
            acc['counts'][key] = 1
        elif np.isfinite(value):
            acc['totals'][key] += value
            acc['counts'][key] += 1
    acc['frames'] += 1
    return mask, True


def _finish_eval(acc):
    metrics = {
        key: (acc['totals'][key] / acc['counts'][key]
              if acc['counts'][key] else float('nan'))
        for key in METRIC_KEYS
    }
    far_pixels = acc['far_pixels']
    metrics.update({
        'far_pixel_count': far_pixels,
        'far_violation_rate': (acc['far_violations'] / far_pixels
                               if far_pixels else float('nan')),
        'far_mean_prediction': (acc['far_prediction_sum'] / far_pixels
                                if far_pixels else float('nan')),
    })
    return metrics, acc['frames']


@torch.no_grad()
def evaluate(model, loader, device, args, split, save_dir=None, num_vis=0):
    model.eval()
    acc = _new_eval_accumulator()
    saved = 0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for sample in loader:
        image = sample['image'].to(device, non_blocking=True).float()
        depth = sample['depth'].to(device, non_blocking=True)[0]
        valid = sample['valid_mask'].to(device, non_blocking=True)[0]
        far = sample['far_mask'].to(device, non_blocking=True)[0]
        pred = model(image)
        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear',
                             align_corners=True)[0, 0]
        mask, usable = _accumulate_prediction(acc, pred, depth, valid, far, args)
        if not usable:
            continue

        if save_dir and (num_vis <= 0 or saved < num_vis):
            save_vis(os.path.join(save_dir, f'{split}_{saved:04d}.png'),
                     image[0], depth, mask, pred, args.dmin, args.dmax,
                     sample.get('display_depth', sample['depth']).to(
                         device, non_blocking=True)[0])
            saved += 1

    return _finish_eval(acc)


def _sample_tensor(sample, key, fallback=None):
    value = sample.get(key, fallback)
    if value is None:
        return None
    return value[0] if value.ndim > 2 and value.shape[0] == 1 else value


def _test_frame_record(sample_index, sample, pred, depth, valid, far, args):
    """Build one CSV row of per-frame test metrics."""
    exact_mask = ((valid == 1) & (depth >= args.min_depth) &
                  (depth <= args.metric_depth_max) & torch.isfinite(pred))
    usable = int(exact_mask.sum()) >= 10
    if usable:
        base_metrics = eval_depth(
            pred[exact_mask], depth[exact_mask], extended=True)
    else:
        base_metrics = {key: float('nan') for key in METRIC_KEYS}
        base_metrics['valid_pixel_count'] = int(exact_mask.sum())

    display_depth = _sample_tensor(sample, 'display_depth', sample['depth']).to(
        pred.device, non_blocking=True)
    display_valid = _sample_tensor(
        sample, 'display_valid_mask', torch.isfinite(display_depth)).to(
            pred.device, non_blocking=True).bool()
    if display_depth.shape != pred.shape:
        display_depth = F.interpolate(
            display_depth[None, None].float(), pred.shape[-2:], mode='nearest'
        )[0, 0]
        display_valid = F.interpolate(
            display_valid[None, None].float(), pred.shape[-2:], mode='nearest'
        )[0, 0] > 0.5

    # Ranges below 15 m use exact metric GT only. Ranges at 15--20 m use
    # complete stored labels, intentionally retaining far=19 and sky=20.
    exact_ranges = [entry for entry in
                    (*TEST_COARSE_RANGES, *TEST_PER_METER_RANGES)
                    if entry[2] <= 15.0]
    full_ranges = [entry for entry in
                   (*TEST_COARSE_RANGES, *TEST_PER_METER_RANGES)
                   if entry[1] >= 15.0]
    range_metrics = eval_mae_ranges(
        pred, depth, exact_mask, exact_ranges)
    range_metrics.update(eval_mae_ranges(
        pred, display_depth, display_valid, full_ranges))

    finite_far = (far == 1) & torch.isfinite(pred)
    far_pixels = int(finite_far.sum())
    far_violation_rate = (
        float((pred[finite_far] < args.far_depth).float().mean().item())
        if far_pixels else float('nan')
    )
    far_mean_prediction = (
        float(pred[finite_far].mean().item())
        if far_pixels else float('nan')
    )
    image_path = sample.get('image_path', '')
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]

    record = {
        'row_type': 'frame',
        'sample_index': sample_index,
        'image_path': str(image_path),
        'test_frame_count': 1,
        **prefixed_metrics('test', base_metrics),
        **prefixed_metrics('test', range_metrics),
        'test_far_pixel_count': far_pixels,
        'test_far_violation_rate': far_violation_rate,
        'test_far_mean_prediction': far_mean_prediction,
        'test_evaluated': int(usable),
    }
    return record, usable


def _write_test_perframe_metrics(records, metric_dir):
    """Write frame rows plus a frame-average summary row at the bottom.

    Metric columns are averaged with equal weight per frame, ignoring NaN
    values for frames where that metric cannot be evaluated. Pixel/count
    columns are totals, not averages.
    """
    if not records:
        return None
    os.makedirs(metric_dir, exist_ok=True)
    output_path = os.path.join(metric_dir, 'test_perframe_metric.csv')
    fields = list(records[0])
    average = {
        'row_type': 'frame_average',
        'sample_index': '',
        'image_path': 'FRAME_AVERAGE_ALL_TEST_FRAMES',
    }
    sum_fields = {
        field for field in fields
        if field.startswith('test_pixels_') or field in {
            'test_frame_count', 'test_valid_pixel_count',
            'test_far_pixel_count', 'test_evaluated'
        }
    }
    for field in fields[3:]:
        values = []
        for record in records:
            try:
                value = float(record[field])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if field in sum_fields:
            average[field] = int(round(sum(values))) if values else 0
        else:
            average[field] = float(np.mean(values)) if values else float('nan')

    with open(output_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        writer.writerow(average)
    return output_path


def _put_panel_label(panel, label):
    cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _representative_cell(sample, pred, record, range_spec, args,
                         panel_size=150):
    name, low, high = range_spec
    depth = sample['depth'].to(pred.device)
    valid = sample['valid_mask'].to(pred.device).bool()
    display_depth = sample.get('display_depth', sample['depth']).to(pred.device)
    display_valid = sample.get(
        'display_valid_mask', torch.isfinite(display_depth)).to(
            pred.device).bool()
    if display_depth.shape != pred.shape:
        display_depth = F.interpolate(
            display_depth[None, None].float(), pred.shape[-2:], mode='nearest'
        )[0, 0]
        display_valid = F.interpolate(
            display_valid[None, None].float(), pred.shape[-2:], mode='nearest'
        )[0, 0] > 0.5

    if high <= 15.0:
        range_mask = (valid & torch.isfinite(pred) &
                      (depth >= low) & (depth < high))
        error_target = depth
    else:
        range_mask = (display_valid & torch.isfinite(pred) &
                      (display_depth >= low) & (display_depth < high))
        error_target = display_depth

    gt_color = _colorize(display_depth, display_valid, args.dmin, args.dmax)
    pred_color = _colorize(
        pred, torch.isfinite(pred), args.dmin, args.dmax)
    abs_error = torch.abs(pred - error_target)
    error_color = _colorize(abs_error, range_mask, 0.0, 5.0)
    input_color = _display_input(sample['image'], gt_color.shape[:2])
    panels = [
        (input_color, 'Input'),
        (gt_color, 'GT'),
        (pred_color, 'Prediction'),
        (error_color, '|error| 0-5m'),
    ]
    resized = []
    for panel, label in panels:
        panel = cv2.resize(panel, (panel_size, panel_size),
                           interpolation=cv2.INTER_AREA)
        _put_panel_label(panel, label)
        resized.append(panel)
    body = np.vstack([np.hstack(resized[:2]), np.hstack(resized[2:])])
    header = np.full((48, body.shape[1], 3), 40, dtype=np.uint8)
    path = os.path.basename(record['image_path'])
    metric = record[f'test_mae_{name}']
    pixels = int(record[f'test_pixels_{name}'])
    cv2.putText(header, path[:42], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f'MAE={metric:.3f} m  pixels={pixels}', (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([header, body])


@torch.no_grad()
def _save_representative_metrics(model, dataset, records, device, args,
                                 metric_dir):
    """Save one five-row, best/median/worst test summary image."""
    selections = []
    for range_spec in TEST_COARSE_RANGES:
        name = range_spec[0]
        metric_key = f'test_mae_{name}'
        pixels_key = f'test_pixels_{name}'
        eligible = [
            record for record in records
            if int(record[pixels_key]) >= args.test_representative_min_pixels
            and np.isfinite(float(record[metric_key]))
        ]
        eligible.sort(key=lambda record: float(record[metric_key]))
        if eligible:
            chosen = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
        else:
            chosen = [None, None, None]
        selections.append((range_spec, chosen))

    selected_indices = sorted({
        int(record['sample_index'])
        for _, chosen in selections for record in chosen if record is not None
    })
    predictions = {}
    samples = {}
    model.eval()
    for index in selected_indices:
        sample = dataset[index]
        image = sample['image'][None].to(device, non_blocking=True).float()
        target_hw = sample['depth'].shape[-2:]
        pred = model(image)
        pred = F.interpolate(
            pred[:, None], target_hw, mode='bilinear', align_corners=True
        )[0, 0]
        samples[index] = sample
        predictions[index] = pred

    cell_h, cell_w = 348, 300
    top_h, label_w = 42, 165
    canvas = np.full(
        (top_h + len(TEST_COARSE_RANGES) * cell_h,
         label_w + 3 * cell_w, 3), 25, dtype=np.uint8)
    for column, title in enumerate(('Best', 'Median', 'Worst')):
        x = label_w + column * cell_w + cell_w // 2 - 35
        cv2.putText(canvas, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)

    for row, (range_spec, chosen) in enumerate(selections):
        name, low, high = range_spec
        y0 = top_h + row * cell_h
        cv2.putText(canvas, f'MAE {low:g}-{min(high, 20):g} m',
                    (8, y0 + cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (255, 255, 255), 1, cv2.LINE_AA)
        for column, record in enumerate(chosen):
            x0 = label_w + column * cell_w
            if record is None:
                cv2.putText(
                    canvas, f'N/A (<{args.test_representative_min_pixels} px)',
                    (x0 + 55, y0 + cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1,
                    cv2.LINE_AA)
                continue
            index = int(record['sample_index'])
            cell = _representative_cell(
                samples[index], predictions[index], record, range_spec, args)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    os.makedirs(metric_dir, exist_ok=True)
    output_path = os.path.join(metric_dir, 'representative_metrics.png')
    cv2.imwrite(output_path, canvas)
    return output_path


@torch.no_grad()
def evaluate_test(model, loader, device, args, save_dir, num_vis, metric_dir):
    """Evaluate test once, save per-frame CSV, then render representatives."""
    model.eval()
    acc = _new_eval_accumulator()
    records = []
    saved = 0
    os.makedirs(save_dir, exist_ok=True)
    for sample_index, sample in enumerate(loader):
        image = sample['image'].to(device, non_blocking=True).float()
        depth = sample['depth'].to(device, non_blocking=True)[0]
        valid = sample['valid_mask'].to(device, non_blocking=True)[0]
        far = sample['far_mask'].to(device, non_blocking=True)[0]
        pred = model(image)
        pred = F.interpolate(
            pred[:, None], depth.shape[-2:], mode='bilinear',
            align_corners=True)[0, 0]
        mask, usable = _accumulate_prediction(acc, pred, depth, valid, far, args)
        record, _ = _test_frame_record(
            sample_index, sample, pred, depth, valid, far, args)
        records.append(record)
        if usable and (num_vis <= 0 or saved < num_vis):
            save_vis(
                os.path.join(save_dir, f'test_{saved:04d}.png'),
                image[0], depth, mask, pred, args.dmin, args.dmax,
                _sample_tensor(sample, 'display_depth', sample['depth']).to(
                    device, non_blocking=True))
            saved += 1

    perframe_path = _write_test_perframe_metrics(records, metric_dir)
    representative_path = _save_representative_metrics(
        model, loader.dataset, records, device, args, metric_dir)
    print(f'[test metrics] per-frame CSV: {perframe_path}')
    print(f'[test metrics] representative figure: {representative_path}')
    return _finish_eval(acc)


def prefixed_metrics(prefix, metrics):
    return {f'{prefix}_{key}': value for key, value in metrics.items()}


def load_test_checkpoint(model, checkpoint_path, args, fallback_epoch=-1):
    """Load and validate the custom checkpoint; legacy checkpoints still work."""
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f'checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    validate_checkpoint_architecture(checkpoint, model, checkpoint_path)
    checkpoint_encoder = checkpoint.get('encoder') if isinstance(checkpoint, dict) else None
    if checkpoint_encoder and checkpoint_encoder != args.encoder:
        raise SystemExit(f'checkpoint encoder={checkpoint_encoder}, but --encoder={args.encoder}')
    checkpoint_size = checkpoint.get('img_size') if isinstance(checkpoint, dict) else None
    if checkpoint_size and int(checkpoint_size) != args.img_size:
        raise SystemExit(f'checkpoint img_size={checkpoint_size}, but --img-size={args.img_size}')
    checkpoint_max = checkpoint.get('max_depth') if isinstance(checkpoint, dict) else None
    if checkpoint_max and float(checkpoint_max) != args.max_depth:
        raise SystemExit(f'checkpoint max_depth={checkpoint_max}, but --max-depth={args.max_depth}')
    checkpoint_input_mode = (checkpoint.get('input_mode')
                             if isinstance(checkpoint, dict) else None)
    if checkpoint_input_mode and checkpoint_input_mode != args.input_mode:
        raise SystemExit(
            f'checkpoint input_mode={checkpoint_input_mode}, but '
            f'--input-mode={args.input_mode}')

    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(_strip_module(state_dict), strict=True)
    epoch = (int(checkpoint.get('epoch', fallback_epoch))
             if isinstance(checkpoint, dict) else fallback_epoch)
    return checkpoint, epoch


def run_test(model, checkpoint_path, test_loader, device, args, vis_root,
             checkpoint_label=None, fallback_epoch=-1, writer=None):
    """Load one checkpoint, evaluate the test split, and record its results."""
    checkpoint, best_epoch = load_test_checkpoint(
        model, checkpoint_path, args, fallback_epoch)
    test_start = time.perf_counter()
    metric_dir = os.path.join(args.save_path, 'metric')
    test_metrics, test_frames = evaluate_test(
        model, test_loader, device, args,
        os.path.join(vis_root, 'test'), args.test_vis_num, metric_dir)
    test_row = {
        'checkpoint': checkpoint_label or os.path.abspath(checkpoint_path),
        'best_epoch': best_epoch,
        **prefixed_metrics('test', test_metrics),
        'test_evaluated_frames': test_frames,
        'test_seconds': time.perf_counter() - test_start,
    }
    append_csv(os.path.join(args.save_path, 'test_metrics.csv'), test_row)
    if writer:
        for key, value in test_metrics.items():
            writer.add_scalar(f'test/{key}', value, best_epoch)
    print(f"[test best epoch={best_epoch}] abs_rel={test_metrics['abs_rel']:.4f} "
          f"mae={test_metrics['mae']:.3f}m rmse={test_metrics['rmse']:.3f}m "
          f"d1={test_metrics['d1']:.4f} frames={test_frames}")
    return test_row


@torch.no_grad()
def run_vkitti_comparison(model, checkpoint_path, test_loader, device, args,
                          vis_root, fallback_epoch=-1, writer=None):
    """Compare models sequentially so both never occupy GPU memory together."""
    # Deliberately lazy: ordinary training/testing does not require this model
    # module or the official checkpoint.
    from model_metric_depth.dpt_vkitti import VKITTIDepthAnythingV2

    if not os.path.isfile(args.vkitti_checkpoint):
        raise SystemExit(f'VKITTI checkpoint not found: {args.vkitti_checkpoint}')
    _, best_epoch = load_test_checkpoint(
        model, checkpoint_path, args, fallback_epoch)
    model.eval()
    custom_acc = _new_eval_accumulator()
    comparison_dir = os.path.join(vis_root, 'test_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    saved = 0
    custom_records = []
    custom_started = time.perf_counter()
    for sample_index, sample in enumerate(test_loader):
        image = sample['image'].to(device, non_blocking=True).float()
        depth = sample['depth'].to(device, non_blocking=True)[0]
        valid = sample['valid_mask'].to(device, non_blocking=True)[0]
        far = sample['far_mask'].to(device, non_blocking=True)[0]
        custom_pred = model(image)
        custom_pred = F.interpolate(
            custom_pred[:, None], depth.shape[-2:], mode='bilinear',
            align_corners=True)[0, 0]
        _, custom_usable = _accumulate_prediction(
            custom_acc, custom_pred, depth, valid, far, args)
        custom_record, _ = _test_frame_record(
            sample_index, sample, custom_pred, depth, valid, far, args)
        custom_records.append(custom_record)
        if (custom_usable and
                (args.test_vis_num <= 0 or saved < args.test_vis_num)):
            display_depth = sample.get('display_depth', sample['depth']).to(
                device, non_blocking=True)[0]
            display_valid = torch.isfinite(display_depth) & (display_depth > 0)
            gt_color = _colorize(
                display_depth, display_valid, args.dmin, args.dmax)
            custom_color = _colorize(
                custom_pred, torch.ones_like(custom_pred, dtype=torch.bool),
                args.dmin, args.dmax)
            input_color = _display_input(image[0], gt_color.shape[:2])
            panels = [(input_color, 'Input image'),
                      (gt_color, 'GT metric depth'),
                      (custom_color, 'Custom prediction')]
            for panel, label in panels:
                cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(
                os.path.join(comparison_dir, f'test_{sample_index:04d}.png'),
                np.hstack([panel for panel, _ in panels]))
            saved += 1
    custom_seconds = time.perf_counter() - custom_started
    custom_metrics, custom_frames = _finish_eval(custom_acc)
    metric_dir = os.path.join(args.save_path, 'metric')
    perframe_path = _write_test_perframe_metrics(custom_records, metric_dir)
    representative_path = _save_representative_metrics(
        model, test_loader.dataset, custom_records, device, args, metric_dir)
    print(f'[test metrics] per-frame CSV: {perframe_path}')
    print(f'[test metrics] representative figure: {representative_path}')

    # Free custom weights before the optional official model enters the GPU.
    model.to('cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()

    vkitti = VKITTIDepthAnythingV2(
        **MODEL_CONFIGS['vits'], max_depth=80.0)
    payload = torch.load(args.vkitti_checkpoint, map_location='cpu')
    state_dict = payload.get('model', payload) if isinstance(payload, dict) else payload
    try:
        vkitti.load_state_dict(_strip_module(state_dict), strict=True)
    except RuntimeError as error:
        raise SystemExit(
            f'VKITTI checkpoint does not match the official metric model: {error}') from error
    vkitti = vkitti.to(device).eval()
    vkitti_acc = _new_eval_accumulator()
    vkitti_started = time.perf_counter()
    for sample_index, sample in enumerate(test_loader):
        image = sample['image'].to(device, non_blocking=True).float()
        depth = sample['depth'].to(device, non_blocking=True)[0]
        valid = sample['valid_mask'].to(device, non_blocking=True)[0]
        far = sample['far_mask'].to(device, non_blocking=True)[0]
        vkitti_pred = vkitti(image)
        vkitti_pred = F.interpolate(
            vkitti_pred[:, None], depth.shape[-2:], mode='bilinear',
            align_corners=True)[0, 0]
        _accumulate_prediction(vkitti_acc, vkitti_pred, depth, valid, far, args)

        comparison_path = os.path.join(
            comparison_dir, f'test_{sample_index:04d}.png')
        prefix = cv2.imread(comparison_path)
        if prefix is not None:
            vkitti_color = _colorize(
                vkitti_pred, torch.ones_like(vkitti_pred, dtype=torch.bool),
                args.dmin, args.dmax)
            cv2.putText(vkitti_color, 'Official VKITTI prediction', (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
            colorbar = _depth_colorbar(
                vkitti_color.shape[0], args.dmin, args.dmax)
            cv2.imwrite(
                comparison_path, np.hstack([prefix, vkitti_color, colorbar]))

    vkitti_seconds = time.perf_counter() - vkitti_started
    vkitti_metrics, vkitti_frames = _finish_eval(vkitti_acc)

    # Preserve the normal custom-model test output used by existing tools.
    custom_row = {
        'checkpoint': 'best.pt',
        'best_epoch': best_epoch,
        **prefixed_metrics('test', custom_metrics),
        'test_evaluated_frames': custom_frames,
        'test_seconds': custom_seconds,
    }
    append_csv(os.path.join(args.save_path, 'test_metrics.csv'), custom_row)

    comparison_path = os.path.join(args.save_path, 'test_model_comparison.csv')
    append_csv(comparison_path, {
        'model': 'custom_best',
        'checkpoint': os.path.abspath(checkpoint_path),
        'model_max_depth': args.max_depth,
        **custom_metrics,
        'evaluated_frames': custom_frames,
        'evaluation_seconds': custom_seconds,
    })
    append_csv(comparison_path, {
        'model': 'official_vkitti',
        'checkpoint': os.path.abspath(args.vkitti_checkpoint),
        'model_max_depth': 80.0,
        **vkitti_metrics,
        'evaluated_frames': vkitti_frames,
        'evaluation_seconds': vkitti_seconds,
    })
    if writer:
        for key, value in custom_metrics.items():
            writer.add_scalar(f'test/{key}', value, best_epoch)
        for key, value in vkitti_metrics.items():
            writer.add_scalar(f'test_vkitti/{key}', value, best_epoch)
    print(f"[test custom epoch={best_epoch}] abs_rel={custom_metrics['abs_rel']:.4f} "
          f"mae={custom_metrics['mae']:.3f}m d1={custom_metrics['d1']:.4f} "
          f"frames={custom_frames}")
    print(f"[test VKITTI] abs_rel={vkitti_metrics['abs_rel']:.4f} "
          f"mae={vkitti_metrics['mae']:.3f}m d1={vkitti_metrics['d1']:.4f} "
          f"frames={vkitti_frames}")
    del vkitti
    if device == 'cuda':
        torch.cuda.empty_cache()
    return custom_row


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(argv=None):
    cli_argv = list(sys.argv[1:] if argv is None else argv)

    def cli_provided(option):
        """True when an option came from the CLI rather than YAML defaults."""
        return any(token == option or token.startswith(option + '=')
                   for token in cli_argv)

    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument('--config')
    config_probe_args, _ = config_probe.parse_known_args(argv)
    input_config, input_config_path = load_training_config(
        config_probe_args.config)

    parser = argparse.ArgumentParser('Train/validate/test fisheye metric DA-V2')
    parser.add_argument('--config', help='input training YAML; CLI overrides YAML')
    # Data
    parser.add_argument('--data', default=None,
                        help='NPZ directory; automatically split into train/val/test')
    parser.add_argument('--train-list', default=None, help='explicit train .txt list')
    parser.add_argument('--val-list', default=None, help='explicit validation .txt list')
    parser.add_argument('--test-list', default=None, help='explicit held-out test .txt list')
    parser.add_argument('--split-blocks', type=int, default=10,
                        help='number of continuous blocks for automatic split')
    parser.add_argument('--target-key', default='depth_aligned',
                        choices=['depth_aligned', 'rs_depth_L'])
    parser.add_argument('--input-mode', choices=['mono', 'color'], default='mono',
                        help='mono converts input to grayscale then repeats it '
                             'over three channels for the pretrained encoder')
    parser.add_argument('--sky-as-far', dest='sky_as_far', action='store_true')
    parser.add_argument('--no-sky-as-far', dest='sky_as_far', action='store_false')
    # Model / optimizer
    parser.add_argument('--encoder', default='vits', choices=MODEL_CONFIGS)
    parser.add_argument('--model-architecture', choices=MODEL_ARCHITECTURES,
                        default='dpt',
                        help='metric model implementation; dpt keeps the '
                             'existing default behavior')
    parser.add_argument('--img-size', type=int, default=322)
    parser.add_argument('--min-depth', type=float, default=0.2)
    parser.add_argument('--metric-depth-max', type=float, default=15.0,
                        help='largest distance trained with exact metric SiLog loss')
    parser.add_argument('--far-depth', type=float, default=19.0,
                        help='minimum acceptable prediction on far-mask pixels')
    parser.add_argument('--max-depth', type=float, default=20.0)
    parser.add_argument('--far-loss-weight', type=float, default=0.1,
                        help='weight of one-sided far loss relative to SiLog')
    parser.add_argument('--loss', choices=['silog', 'range-weighted-silog'],
                        default='silog',
                        help='metric loss; silog preserves the original behavior')
    parser.add_argument('--silog-0-2-weight', type=float, default=0.4)
    parser.add_argument('--silog-2-5-weight', type=float, default=0.3)
    parser.add_argument('--silog-5-10-weight', type=float, default=0.2)
    parser.add_argument('--silog-10-20-weight', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--pretrained-from', default=None)
    parser.add_argument('--load-head', dest='load_head', action='store_true')
    parser.add_argument('--no-load-head', dest='load_head', action='store_false')
    parser.add_argument('--resume', default=None,
                        help='resume from last.pt (optimizer/scaler state required)')
    parser.add_argument('--model-training', type=parse_bool, default=True,
                        help='true: train/validate/test; false: test inference only')
    parser.add_argument('--checkpoint', default=None,
                        help='best.pt used when --model-training false')
    parser.add_argument('--compare-vkitti', dest='compare_vkitti',
                        action='store_true',
                        help='compare test results with the official VKITTI vits model')
    parser.add_argument('--no-compare-vkitti', dest='compare_vkitti',
                        action='store_false')
    parser.add_argument('--vkitti-checkpoint', default=None,
                        help='official VKITTI vits .pth; required only with '
                             '--compare-vkitti')
    # Output / visualization
    parser.add_argument('--save-path')
    parser.add_argument('--verbose-vis', dest='verbose_vis', action='store_true',
                        help='save validation figures at regular epoch intervals')
    parser.add_argument('--no-verbose-vis', dest='verbose_vis', action='store_false')
    parser.add_argument('--vis-epoch-interval', '--vis-every', dest='vis_epoch_interval',
                        type=int, default=5)
    parser.add_argument('--valid-vis-num', '--num-vis', dest='valid_vis_num',
                        type=int, default=6, help='validation figures to save; 0 means all')
    parser.add_argument('--test-vis-num', type=int, default=0,
                        help='test figures to save after training; 0 means all')
    parser.add_argument('--test-representative-min-pixels', type=int, default=100,
                        help='minimum GT pixels in a distance range for a frame '
                             'to participate in best/median/worst selection')
    parser.add_argument('--dmin', type=float, default=0.2,
                        help='shared visualization/colorbar minimum only')
    parser.add_argument('--dmax', type=float, default=20.0,
                        help='shared visualization/colorbar maximum only')
    parser.add_argument('--no-tb', action='store_true')
    parser.add_argument('--tb', dest='no_tb', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    # Paired true/false flags otherwise inherit the last argparse action's
    # default. Keep the historical no-config behavior explicit here.
    parser.set_defaults(sky_as_far=False, amp=False, load_head=False,
                        verbose_vis=False, no_tb=False, compare_vkitti=False)
    valid_config_keys = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(input_config) - valid_config_keys)
    if unknown_config_keys:
        parser.error(f'unknown key(s) in --config: {unknown_config_keys}')
    parser.set_defaults(**input_config)
    args = parser.parse_args(argv)
    args.config = input_config_path

    if not args.model_training:
        # A saved training run config naturally records its original dataset
        # and train/validation lists.  They are provenance, not requests to
        # train during a later inference-only reuse.  Clear only values that
        # were inherited from YAML; explicitly passing any of these options on
        # the command line remains an error below.
        for option, dest in (
                ('--data', 'data'),
                ('--train-list', 'train_list'),
                ('--val-list', 'val_list')):
            if not cli_provided(option):
                setattr(args, dest, None)

    if not args.save_path:
        parser.error('--save-path is required in YAML or CLI')

    if args.vis_epoch_interval <= 0:
        parser.error('--vis-epoch-interval must be positive')
    if args.test_representative_min_pixels <= 0:
        parser.error('--test-representative-min-pixels must be positive')
    if not args.min_depth < args.metric_depth_max < args.far_depth < args.max_depth:
        parser.error('require min-depth < metric-depth-max < far-depth < max-depth')
    if args.far_loss_weight < 0:
        parser.error('--far-loss-weight must be non-negative')
    if args.compare_vkitti and not args.vkitti_checkpoint:
        parser.error('--compare-vkitti requires --vkitti-checkpoint')
    range_weights = (
        args.silog_0_2_weight, args.silog_2_5_weight,
        args.silog_5_10_weight, args.silog_10_20_weight,
    )
    if any(weight < 0 for weight in range_weights):
        parser.error('all range SiLog weights must be non-negative')
    explicit = [args.train_list, args.val_list, args.test_list]
    if args.model_training and any(explicit) and not all(explicit):
        parser.error('provide all of --train-list, --val-list, and --test-list')
    if not args.model_training:
        if not args.test_list:
            parser.error('--model-training false requires --test-list')
        if not args.checkpoint:
            parser.error('--model-training false requires --checkpoint')
        if args.train_list or args.val_list or args.data:
            parser.error('--model-training false accepts only --test-list, not training data')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    metric_dir = os.path.join(args.save_path, 'metric')
    os.makedirs(metric_dir, exist_ok=True)
    vis_root = os.path.join(args.save_path, 'vis')
    os.makedirs(vis_root, exist_ok=True)
    split_dir = os.path.join(args.save_path, 'split')
    os.makedirs(split_dir, exist_ok=True)
    stored_lists = {
        name: os.path.join(split_dir, f'{name}.txt')
        for name in ('train', 'val', 'test')
    }

    if not args.model_training:
        train_files, val_files = [], []
        test_files = read_list(args.test_list)
        split_details = {'method': 'test_only_list'}
    elif all(explicit):
        train_files = read_list(args.train_list)
        val_files = read_list(args.val_list)
        test_files = read_list(args.test_list)
        split_details = {'method': 'explicit_lists'}
    elif all(os.path.isfile(path) for path in stored_lists.values()):
        train_files = read_list(stored_lists['train'])
        val_files = read_list(stored_lists['val'])
        test_files = read_list(stored_lists['test'])
        split_details = {'method': 'reused_saved_lists'}
        print(f'[split] reusing {split_dir}')
    elif any(os.path.exists(path) for path in stored_lists.values()):
        raise SystemExit(f'incomplete saved split in {split_dir}; expected train.txt, '
                         'val.txt, and test.txt')
    else:
        if not args.data:
            parser.error('pass --data, all three explicit lists, or reuse save-path/split')
        files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
        if not files:
            raise SystemExit(f'no .npz files in {args.data}')
        train_files, val_files, test_files, split_details = contiguous_block_split(
            files, args.split_blocks, args.seed)

    split_sets = [set(train_files), set(val_files), set(test_files)]
    if args.model_training:
        if (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2]
                or split_sets[1] & split_sets[2]):
            raise SystemExit('train/validation/test lists overlap')
        if not all(split_sets):
            raise SystemExit('train, validation, and test splits must all be non-empty')
        if split_details['method'] != 'reused_saved_lists':
            write_list(train_files, stored_lists['train'])
            write_list(val_files, stored_lists['val'])
    elif not test_files:
        raise SystemExit('test split must be non-empty')
    write_list(test_files, stored_lists['test'])
    train_list = stored_lists['train']
    val_list = stored_lists['val']
    test_list = stored_lists['test']
    counts = {'train': len(train_files), 'validation': len(val_files), 'test': len(test_files)}
    save_run_metadata(args, counts, split_details)
    print(f"[split] train={counts['train']} val={counts['validation']} "
          f"test={counts['test']} details={split_details}")

    size = (args.img_size, args.img_size)
    dataset_args = dict(size=size, target_key=args.target_key,
                        input_mode=args.input_mode,
                        min_depth=args.min_depth,
                        max_depth=args.metric_depth_max,
                        sky_as_far=args.sky_as_far,
                        far_depth=args.far_depth)
    # Validation mode is deterministic, so it is also correct for test data.
    test_ds = FisheyeNPZ(test_list, 'val', **dataset_args)
    pin_memory = device == 'cuda'
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=pin_memory)
    if args.model_training:
        train_ds = FisheyeNPZ(train_list, 'train', **dataset_args)
        val_ds = FisheyeNPZ(val_list, 'val', **dataset_args)
        train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                                  drop_last=True, num_workers=args.workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, pin_memory=pin_memory)
    print(f'[data] device={device} encoder={args.encoder} img={args.img_size} '
          f'target={args.target_key} input={args.input_mode}')

    model = build_model(args)
    if args.pretrained_from:
        load_pretrained(model, args.pretrained_from, args.load_head)
    model = model.to(device)

    if not args.model_training:
        if args.compare_vkitti:
            run_vkitti_comparison(
                model, args.checkpoint, test_loader, device, args, vis_root)
        else:
            run_test(model, args.checkpoint, test_loader, device, args, vis_root)
        print(f'done. outputs: {args.save_path}')
        return

    optimizer = AdamW(
        [{'params': [p for n, p in model.named_parameters() if 'pretrained' in n],
          'lr': args.lr},
         {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n],
          'lr': args.lr * HEAD_LR_MULTIPLIER}],
        lr=args.lr, betas=OPTIMIZER_BETAS,
        weight_decay=OPTIMIZER_WEIGHT_DECAY)
    criterion = SiLogLoss(lambd=SILOG_LAMBDA).to(device)
    range_criterion = RangeWeightedSiLogLoss(
        lambd=SILOG_LAMBDA, weights=range_weights).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')

    start_epoch, best, best_epoch = 0, float('inf'), -1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise SystemExit(f'resume checkpoint not found: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        validate_checkpoint_architecture(checkpoint, model, args.resume)
        checkpoint_input_mode = checkpoint.get('input_mode')
        if checkpoint_input_mode and checkpoint_input_mode != args.input_mode:
            raise SystemExit(
                f'resume input_mode={checkpoint_input_mode}, but '
                f'--input-mode={args.input_mode}')
        if 'optimizer' not in checkpoint:
            raise SystemExit('--resume requires last.pt, not best.pt')
        model.load_state_dict(_strip_module(checkpoint['model']), strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        best = float(checkpoint.get('best', float('inf')))
        best_epoch = int(checkpoint.get('best_epoch', -1))
        print(f'[resume] epoch={start_epoch} best_abs_rel={best:.6f}')

    writer = SummaryWriter(args.save_path) if (_HAS_TB and not args.no_tb) else None
    total_iterations = args.epochs * max(len(train_loader), 1)
    # Keep the historical root CSV for existing summary tools, and write the
    # requested canonical copy under metric/.
    metrics_csv = os.path.join(args.save_path, 'metrics.csv')
    nested_metrics_csv = os.path.join(metric_dir, 'metric.csv')
    best_path = os.path.join(args.save_path, 'best.pt')

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_total_loss = 0.0
        running_metric_loss = 0.0
        running_global_silog = 0.0
        running_far_loss = 0.0
        running_range_silog = {
            name: 0.0 for name, _, _ in RangeWeightedSiLogLoss.RANGES
        }
        batches = 0
        for iteration, sample in enumerate(train_loader):
            image = sample['image'].to(device, non_blocking=True)
            depth = sample['depth'].to(device, non_blocking=True)
            valid = sample['valid_mask'].to(device, non_blocking=True)
            far = sample['far_mask'].to(device, non_blocking=True)
            if random.random() < 0.5:
                image = image.flip(-1)
                depth = depth.flip(-1)
                valid = valid.flip(-1)
                far = far.flip(-1)

            mask = ((valid == 1) & (depth >= args.min_depth) &
                    (depth <= args.metric_depth_max))
            if mask.sum() < 10 and not far.any():
                continue
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp and device == 'cuda'):
                prediction = model(image)
                if mask.sum() >= 10:
                    if args.loss == 'range-weighted-silog':
                        metric_loss, silog_parts = range_criterion(
                            prediction, depth, mask, return_components=True)
                    else:
                        metric_loss = criterion(prediction, depth, mask)
                        with torch.no_grad():
                            _, silog_parts = range_criterion(
                                prediction, depth, mask, return_components=True)
                else:
                    metric_loss = prediction.sum() * 0.0
                    silog_parts = {
                        name: prediction.sum() * 0.0
                        for name, _, _ in RangeWeightedSiLogLoss.RANGES
                    }
                    silog_parts['global'] = prediction.sum() * 0.0
                far_loss = (F.relu(args.far_depth - prediction[far]).mean()
                            if far.any() else prediction.sum() * 0.0)
                loss = metric_loss + args.far_loss_weight * far_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_iteration = epoch * len(train_loader) + iteration
            lr = (args.lr *
                  (1 - global_iteration / total_iterations) ** LR_SCHEDULE_POWER)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * HEAD_LR_MULTIPLIER
            running_total_loss += loss.item()
            running_metric_loss += metric_loss.item()
            running_global_silog += silog_parts['global'].item()
            running_far_loss += far_loss.item()
            for name in running_range_silog:
                running_range_silog[name] += silog_parts[name].item()
            batches += 1
            if writer and global_iteration % 20 == 0:
                writer.add_scalar('train/loss_total', loss.item(), global_iteration)
                writer.add_scalar('train/loss_metric', metric_loss.item(), global_iteration)
                writer.add_scalar('train/loss_silog_global',
                                  silog_parts['global'].item(), global_iteration)
                writer.add_scalar('train/loss_far', far_loss.item(), global_iteration)
                for name in running_range_silog:
                    writer.add_scalar(f'train/silog_{name}',
                                      silog_parts[name].item(), global_iteration)
                writer.add_scalar('train/lr_encoder', lr, global_iteration)
            if iteration % 50 == 0:
                print(f'  ep{epoch:03d} it{iteration:04d}/{len(train_loader)} '
                      f'lr={lr:.2e} total={loss.item():.4f} '
                      f'metric={metric_loss.item():.4f} far={far_loss.item():.4f}')

        is_last_epoch = epoch == args.epochs - 1
        periodic_vis = args.verbose_vis and epoch % args.vis_epoch_interval == 0
        val_vis_dir = None
        if periodic_vis:
            val_vis_dir = os.path.join(vis_root, 'val', f'epoch_{epoch:04d}')
        elif is_last_epoch:
            val_vis_dir = os.path.join(vis_root, 'val', 'last')
        val_metrics, val_frames = evaluate(
            model, val_loader, device, args, 'val', val_vis_dir, args.valid_vis_num)
        train_total_loss = running_total_loss / max(batches, 1)
        train_metric_loss = running_metric_loss / max(batches, 1)
        train_global_silog = running_global_silog / max(batches, 1)
        train_far_loss = running_far_loss / max(batches, 1)
        train_range_silog = {
            name: value / max(batches, 1)
            for name, value in running_range_silog.items()
        }
        is_best = np.isfinite(val_metrics['abs_rel']) and val_metrics['abs_rel'] < best
        if is_best:
            best = val_metrics['abs_rel']
            best_epoch = epoch
            save_best(best_path, model, epoch, best, args)
            print(f'  -> new best abs_rel={best:.6f}')

        save_last(os.path.join(args.save_path, 'last.pt'), model, optimizer, scaler,
                  epoch, best, best_epoch, args)
        row = {
            'epoch': epoch,
            'train_total_loss': train_total_loss,
            'train_metric_loss': train_metric_loss,
            'train_silog_loss': train_global_silog,
            'train_far_loss': train_far_loss,
            **{f'train_silog_{name}': value
               for name, value in train_range_silog.items()},
            **prefixed_metrics('val', val_metrics),
            'val_evaluated_frames': val_frames,
            'lr_encoder': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'epoch_seconds': time.perf_counter() - epoch_start,
            'is_best': is_best,
        }
        append_csv(metrics_csv, row)
        append_csv(nested_metrics_csv, row)
        print(f'[epoch {epoch:03d}] total={train_total_loss:.4f} '
              f'metric={train_metric_loss:.4f} far={train_far_loss:.4f} '
              f"val_abs_rel={val_metrics['abs_rel']:.4f} "
              f"val_mae={val_metrics['mae']:.3f}m "
              f"val_rmse={val_metrics['rmse']:.3f}m d1={val_metrics['d1']:.4f}")
        if writer:
            writer.add_scalar('train/loss_total_epoch', train_total_loss, epoch)
            writer.add_scalar('train/loss_metric_epoch', train_metric_loss, epoch)
            writer.add_scalar('train/loss_silog_global_epoch', train_global_silog, epoch)
            writer.add_scalar('train/loss_far_epoch', train_far_loss, epoch)
            for name, value in train_range_silog.items():
                writer.add_scalar(f'train/silog_{name}_epoch', value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)

    if not os.path.isfile(best_path):
        raise SystemExit('no best.pt was produced; validation had no usable frames')

    if args.compare_vkitti:
        run_vkitti_comparison(
            model, best_path, test_loader, device, args, vis_root,
            fallback_epoch=best_epoch, writer=writer)
    else:
        run_test(model, best_path, test_loader, device, args, vis_root,
                 checkpoint_label='best.pt', fallback_epoch=best_epoch,
                 writer=writer)
    if writer:
        writer.close()
    print(f'done. outputs: {args.save_path}')


if __name__ == '__main__':
    main()
