#!/usr/bin/env python3
"""

Offline metric depth from ROS 2 images; no ROS runtime or bag playback needed.

conda activate aiseed1
cd ~/Projects/da_V2/metric_depth

python infer_fisheye_rosbag.py \
  --bag /path/to/bag \
  --topic /your/left/image_topic \
  --checkpoint runs/fisheye_vits_20260918_1138/best.pt \
  --outdir rosbag_predictions
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

IMAGE_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')


def decode_image(msg, msgtype):
    """Decode supported camera encodings to uint8 BGR, respecting row padding."""
    if msgtype == IMAGE_TYPES[1]:
        if 'compressedDepth' in msg.format:
            raise ValueError('compressedDepth is a depth topic, not a camera input')
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode compressed camera image')
        return image
    encoding = msg.encoding.lower()
    channels = {'mono8': 1, '8uc1': 1, 'bgr8': 3, 'rgb8': 3,
                'bgra8': 4, 'rgba8': 4}.get(encoding)
    if channels is None:
        raise ValueError(f'Unsupported camera encoding: {msg.encoding}; use an 8-bit image topic')
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if h <= 0 or w <= 0 or step < w * channels:
        raise ValueError('Invalid image dimensions/step')
    rows = np.frombuffer(msg.data, np.uint8).reshape(h, step)
    image = rows[:, :w * channels].reshape(h, w, channels)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(image[..., :3])


def decode_depth(msg, scale=0.001):
    """Convert RealSense 16-bit depth (scaled units) or float depth (meters)."""
    encoding = msg.encoding.lower()
    kind = {'16uc1': 'u2', 'mono16': 'u2', '32fc1': 'f4'}.get(encoding)
    if kind is None:
        raise ValueError(f'Unsupported D455 depth encoding: {msg.encoding}')
    dtype = np.dtype(('>' if msg.is_bigendian else '<') + kind)
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    depth = rows[:, :msg.width * dtype.itemsize].copy().view(dtype).reshape(msg.height, msg.width).astype(np.float32)
    if kind == 'u2':
        depth *= scale
    return depth


class NearestDepth:
    """Two-frame streaming buffer, matched using bag recording timestamps."""
    def __init__(self, reader, connections):
        self.reader = reader
        self.messages = iter(reader.messages(connections=connections))
        self.previous = None
        self.following = next(self.messages, None)

    def get(self, timestamp, tolerance):
        while self.following is not None and self.following[1] < timestamp:
            self.previous = self.following
            self.following = next(self.messages, None)
        candidates = [x for x in (self.previous, self.following) if x is not None]
        if not candidates:
            return None, None
        conn, stamp, raw = min(candidates, key=lambda x: abs(x[1] - timestamp))
        if abs(stamp - timestamp) > tolerance:
            return None, None
        return self.reader.deserialize(raw, conn.msgtype), stamp


def depth_color(depth, low, high):
    normalized = np.nan_to_num((depth - low) / (high - low))
    color = cv2.applyColorMap((np.clip(normalized, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~np.isfinite(depth) | (depth <= 0)] = 0
    return color


def labeled_panel(image, label, shape):
    h, w = shape
    scale = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((h + 40, w, 3), np.uint8)
    y, x = 40 + (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(panel, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, min(0.6, w / 800), (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def load_model(args):
    import torch
    from train_fisheye_with_test import (
        MODEL_ARCHITECTURES, build_model, validate_checkpoint_architecture, _strip_module,
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint.get('model_config', {})
    architecture = args.model_architecture
    if architecture is None:
        architecture_id = config.get('architecture_id')
        architecture = next((name for name, info in MODEL_ARCHITECTURES.items()
                             if info['architecture_id'] == architecture_id), None)
        if architecture_id and architecture is None:
            raise ValueError(f'Unknown checkpoint architecture: {architecture_id}')
        architecture = architecture or 'dpt'
    resolved = {'model_architecture': architecture}
    for key in ('encoder', 'img_size', 'max_depth'):
        saved = checkpoint.get(key, config.get(key))
        supplied = getattr(args, key)
        if supplied is not None and saved is not None and supplied != saved:
            raise ValueError(f'--{key.replace("_", "-")}={supplied} conflicts with checkpoint {saved}')
        value = saved if supplied is None else supplied
        if value is None:
            raise ValueError(f'Legacy checkpoint needs --{key.replace("_", "-")}')
        resolved[key] = value
    if resolved['img_size'] <= 0 or resolved['max_depth'] <= 0:
        raise ValueError('Image size and max depth must be positive')
    state = _strip_module(checkpoint.get('model', checkpoint))
    legacy_activation = getattr(args, 'legacy_output_activation', None)
    flat_head = 'depth_head.scratch.output_conv2.0.weight' in state
    if flat_head and legacy_activation is None:
        raise ValueError(
            'This checkpoint has the legacy flat output head, whereas the current '
            'model uses a nested custom head. load_head is only a training initialization '
            'setting. For a legacy Conv-ReLU-Conv head, specify '
            '--legacy-output-activation sigmoid or softplus to match the original '
            'training code. The activation cannot be inferred from checkpoint weights.')
    if legacy_activation and (not flat_head or config.get('architecture_id')):
        raise ValueError('--legacy-output-activation is only for flat-head checkpoints without architecture metadata')
    model = build_model(SimpleNamespace(**resolved))
    if legacy_activation:
        if architecture != 'dpt':
            raise ValueError('Legacy flat-head compatibility requires --model-architecture dpt')
        from model_metric_depth.dpt import DPTHead
        from train_fisheye_with_test import MODEL_CONFIGS
        cfg = MODEL_CONFIGS[resolved['encoder']]
        model.depth_head = DPTHead(model.pretrained.embed_dim, cfg['features'],
                                   out_channels=cfg['out_channels'])
        model.depth_head.scratch.output_conv2[-1] = (
            torch.nn.Sigmoid() if legacy_activation == 'sigmoid' else torch.nn.Softplus())
        resolved['legacy_output_activation'] = legacy_activation
        resolved['head_layout'] = 'flat_conv_relu_conv'
    validate_checkpoint_architecture(checkpoint, model)
    model.load_state_dict(state, strict=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, device, resolved


def make_transform(size):
    from dataset.transform import Resize, NormalizeImage, PrepareForNet
    transforms = [Resize(width=size, height=size, resize_target=False,
                         keep_aspect_ratio=True, ensure_multiple_of=14,
                         resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
                  NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  PrepareForNet()]

    def transform(bgr):
        sample = {'image': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0}
        for operation in transforms:
            sample = operation(sample)
        return sample['image']
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path, help='ROS 2 bag directory or standalone .mcap')
    parser.add_argument('--rectify-calibration', type=Path, help='Double-sphere multi-pair YAML for raw fisheye images')
    parser.add_argument('--rectify-pair', default='2_1', help='Left/right camera pair, e.g. 2_1')
    parser.add_argument('--rectify-size', type=int, default=320)
    parser.add_argument('--rectify-fov', type=float, default=90.0)
    parser.add_argument('--input-rectified', action='store_true', help='Explicitly confirm input already rectified')
    parser.add_argument('--topic', help='Left camera topic used to generate training NPZs')
    parser.add_argument('--list-topics', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--outdir', type=Path, default=Path('rosbag_depth'))
    parser.add_argument('--encoder', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--model-architecture', choices=['dpt', 'dpt-custom'])
    parser.add_argument('--legacy-output-activation', choices=['sigmoid', 'softplus'],
                        help='For metadata-free legacy Conv-ReLU-Conv heads only: must match training code')
    parser.add_argument('--img-size', type=int)
    parser.add_argument('--max-depth', type=float)
    parser.add_argument('--device', help='e.g. cpu, cuda, cuda:1; defaults to CUDA if available')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, help='Maximum number of predictions')
    parser.add_argument('--start-seconds', type=float, default=0, help='Offset from bag start')
    parser.add_argument('--duration', type=float, help='Seconds to read after start offset')
    parser.add_argument('--vis-min', type=float, default=0.2)
    parser.add_argument('--vis-max', type=float, default=20.0)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--d455-topic', default='/d455/d455_node/depth/image_rect_raw',
                        help='Optional reference depth panel; omitted if topic absent')
    parser.add_argument('--d455-depth-scale', type=float, default=0.001,
                        help='Meters per integer D455 depth unit (float depth is already meters)')
    parser.add_argument('--sync-tolerance', type=float, default=0.1,
                        help='Maximum bag timestamp difference in seconds')
    args = parser.parse_args()
    if args.sync_tolerance < 0 or args.d455_depth_scale <= 0:
        parser.error('sync-tolerance must be nonnegative and depth-scale positive')
    if args.frame_stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('frame-stride and max-frames must be positive')
    if args.start_seconds < 0 or (args.duration is not None and args.duration <= 0):
        parser.error('start-seconds must be nonnegative and duration positive')
    if not args.vis_max > args.vis_min:
        parser.error('vis-max must exceed vis-min')
    if not args.list_topics and (not args.topic or not args.checkpoint):
        parser.error('--topic and --checkpoint are required for inference')
    rectifier = None
    if not args.list_topics:
        if args.rectify_calibration and args.input_rectified:
            parser.error('Choose rectification or already-rectified input, not both')
        if args.rectify_calibration:
            from fisheye_rectification import DoubleSphereRectifier
            expected_topic = f'/camera_{args.rectify_pair.split("_")[0]}/image_raw'
            if args.topic != expected_topic:
                parser.error(f'Rectification pair expects left input {expected_topic}')
            rectifier = DoubleSphereRectifier(args.rectify_calibration, args.rectify_pair,
                                              args.rectify_size, args.rectify_fov)
        elif not args.input_rectified and 'image_rect' not in args.topic:
            parser.error('Raw camera input requires --rectify-calibration; use --input-rectified only for already rectified images')
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        parser.exit(1, 'Install the offline bag reader in this Python environment: pip install rosbags\n')
    if args.bag.is_dir() and not (args.bag / 'metadata.yaml').exists():
        candidates = sorted(args.bag.glob('*_recovered.mcap')) or sorted(args.bag.glob('*.mcap'))
        if len(candidates) != 1:
            parser.error('Directory has no metadata.yaml; specify one MCAP file explicitly')
        args.bag = candidates[0]
        print(f'[bag] {args.bag}', flush=True)
    with AnyReader([args.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader, \
            AnyReader([args.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as depth_reader:
        if args.list_topics:
            for conn in reader.connections:
                print(f'{conn.topic}\t{conn.msgtype}\t{conn.msgcount} messages')
            return
        connections = [c for c in reader.connections if c.topic == args.topic]
        if not connections or any(c.msgtype not in IMAGE_TYPES for c in connections):
            parser.error('Topic missing or not sensor_msgs Image/CompressedImage; use --list-topics')
        depth_connections = [c for c in depth_reader.connections if c.topic == args.d455_topic]
        if any(c.msgtype != IMAGE_TYPES[0] for c in depth_connections):
            parser.error('D455 reference must be a raw sensor_msgs/Image depth topic')
        reference = NearestDepth(depth_reader, depth_connections) if depth_connections and not args.no_preview else None
        print(f'[reference] {args.d455_topic if reference else "no D455 preview"}', flush=True)
        import torch
        import torch.nn.functional as F
        model, device, resolved = load_model(args)
        transform = make_transform(resolved['img_size'])
        args.outdir.mkdir(parents=True, exist_ok=False)
        (args.outdir / 'depth').mkdir()
        if rectifier:
            (args.outdir / 'rectified').mkdir()
            (args.outdir / 'rectification.json').write_text(json.dumps(rectifier.metadata, indent=2) + '\n')
            np.save(args.outdir / 'rectification_valid_mask.npy', rectifier.valid)
        if not args.no_preview:
            (args.outdir / 'preview').mkdir()
        metadata = {**vars(args), **resolved, 'device': str(device), 'depth_units': 'meters'}
        (args.outdir / 'config.json').write_text(json.dumps(metadata, default=str, indent=2) + '\n')
        start = reader.start_time + round(args.start_seconds * 1e9)
        stop = None if args.duration is None else start + round(args.duration * 1e9)
        count = 0
        with (args.outdir / 'frames.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['index', 'bag_timestamp_ns', 'header_timestamp_ns', 'frame_id', 'depth_file', 'preview_file', 'd455_bag_timestamp_ns', 'd455_delta_ms'])
            with torch.inference_mode():
                for index, (conn, timestamp, raw) in enumerate(reader.messages(connections=connections, start=start, stop=stop)):
                    if index % args.frame_stride:
                        continue
                    msg = reader.deserialize(raw, conn.msgtype)
                    bgr = decode_image(msg, conn.msgtype)
                    if rectifier:
                        bgr = rectifier(bgr)
                    tensor = torch.from_numpy(transform(bgr)).unsqueeze(0).to(device)
                    prediction = model(tensor)
                    depth = F.interpolate(prediction[:, None], bgr.shape[:2], mode='bilinear',
                                          align_corners=True)[0, 0].cpu().numpy().astype(np.float32)
                    stem = f'{count:06d}_{timestamp}'
                    if rectifier:
                        depth[~rectifier.valid] = 0
                        if not cv2.imwrite(str(args.outdir / 'rectified' / f'{stem}.png'), bgr):
                            raise OSError('Failed to save rectified image')
                    depth_file = f'depth/{stem}.npy'
                    np.save(args.outdir / depth_file, depth)
                    preview_file = ''
                    ref_stamp = None
                    if not args.no_preview:
                        shape = bgr.shape[:2]
                        panels = [labeled_panel(bgr, 'Rectified camera' if rectifier else 'Camera input', shape),
                                  labeled_panel(depth_color(depth, args.vis_min, args.vis_max),
                                                f'Prediction ({args.vis_min:g}-{args.vis_max:g} m)', shape)]
                        if reference:
                            ref_msg, ref_stamp = reference.get(timestamp, round(args.sync_tolerance * 1e9))
                            if ref_msg is None:
                                ref_color = np.zeros_like(bgr)
                                label = 'D455: no synchronized frame'
                            else:
                                ref_depth = decode_depth(ref_msg, args.d455_depth_scale)
                                ref_color = depth_color(ref_depth, args.vis_min, args.vis_max)
                                label = f'D455 native view ({args.vis_min:g}-{args.vis_max:g} m)'
                            panels.append(labeled_panel(ref_color, label, shape))
                        preview_file = f'preview/{stem}.jpg'
                        if not cv2.imwrite(str(args.outdir / preview_file), np.concatenate(panels, axis=1)):
                            raise OSError(f'Failed to write {preview_file}')
                    stamp = msg.header.stamp
                    writer.writerow([count, timestamp, int(stamp.sec) * 10**9 + int(stamp.nanosec),
                                     msg.header.frame_id, depth_file, preview_file,
                                     ref_stamp if ref_stamp is not None else '',
                                     (ref_stamp - timestamp) / 1e6 if ref_stamp is not None else ''])
                    stream.flush()
                    count += 1
                    if count == 1 or count % 50 == 0:
                        print(f'[inference] {count} frames saved', flush=True)
                    if args.max_frames is not None and count >= args.max_frames:
                        break
        print(f'Saved {count} depth maps to {args.outdir}')
        if count == 0:
            raise SystemExit('No camera frames found in the selected time interval')


if __name__ == '__main__':
    main()
