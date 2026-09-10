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
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.fisheye_npz import FisheyeNPZ
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth

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

OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_WEIGHT_DECAY = 0.001
HEAD_LR_MULTIPLIER = 10.0
LR_SCHEDULE_POWER = 0.9
SILOG_LAMBDA = 0.5

METRIC_KEYS = [
    'd1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog',
    'mae', 'mae_0.2_2m', 'mae_2_5m', 'mae_5_10m', 'mae_10_20m',
    'median_relative_error', 'valid_pixel_count',
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


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
    config['loss'] = {
        'name': 'SiLogLoss',
        'lambda': SILOG_LAMBDA,
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
    return {
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
    }


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


def save_vis(path, image, depth_gt, valid, pred, dmin, dmax, far_mask=None,
             far_depth=19.0):
    display_gt = depth_gt.clone()
    display_valid = valid.clone()
    if far_mask is not None:
        display_gt[far_mask] = far_depth
        display_valid |= far_mask
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


@torch.no_grad()
def evaluate(model, loader, device, args, split, save_dir=None, num_vis=0):
    model.eval()
    totals = {key: 0.0 for key in METRIC_KEYS}
    counts = {key: 0 for key in METRIC_KEYS}
    evaluated_frames = 0
    far_pixels = 0
    far_violations = 0
    far_prediction_sum = 0.0
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
        finite_far = (far == 1) & torch.isfinite(pred)
        if finite_far.any():
            far_values = pred[finite_far]
            far_pixels += int(finite_far.sum())
            far_violations += int((far_values < args.far_depth).sum())
            far_prediction_sum += float(far_values.sum())
        mask = ((valid == 1) & (depth >= args.min_depth) &
                (depth <= args.metric_depth_max) & torch.isfinite(pred))
        if mask.sum() < 10:
            continue

        result = eval_depth(pred[mask], depth[mask], extended=True)
        for key in METRIC_KEYS:
            value = result[key]
            if key == 'valid_pixel_count':
                totals[key] += value
                counts[key] = 1
            elif np.isfinite(value):
                totals[key] += value
                counts[key] += 1
        evaluated_frames += 1

        if save_dir and (num_vis <= 0 or saved < num_vis):
            save_vis(os.path.join(save_dir, f'{split}_{saved:04d}.png'),
                     image[0], depth, mask, pred, args.dmin, args.dmax, far,
                     args.far_depth)
            saved += 1

    metrics = {
        key: (totals[key] / counts[key] if counts[key] else float('nan'))
        for key in METRIC_KEYS
    }
    metrics.update({
        'far_pixel_count': far_pixels,
        'far_violation_rate': (far_violations / far_pixels
                               if far_pixels else float('nan')),
        'far_mean_prediction': (far_prediction_sum / far_pixels
                                if far_pixels else float('nan')),
    })
    return metrics, evaluated_frames


def prefixed_metrics(prefix, metrics):
    return {f'{prefix}_{key}': value for key, value in metrics.items()}


def run_test(model, checkpoint_path, test_loader, device, args, vis_root,
             checkpoint_label=None, fallback_epoch=-1, writer=None):
    """Load one checkpoint, evaluate the test split, and record its results."""
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f'checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_encoder = checkpoint.get('encoder') if isinstance(checkpoint, dict) else None
    if checkpoint_encoder and checkpoint_encoder != args.encoder:
        raise SystemExit(f'checkpoint encoder={checkpoint_encoder}, but --encoder={args.encoder}')
    checkpoint_size = checkpoint.get('img_size') if isinstance(checkpoint, dict) else None
    if checkpoint_size and int(checkpoint_size) != args.img_size:
        raise SystemExit(f'checkpoint img_size={checkpoint_size}, but --img-size={args.img_size}')
    checkpoint_max = checkpoint.get('max_depth') if isinstance(checkpoint, dict) else None
    if checkpoint_max and float(checkpoint_max) != args.max_depth:
        raise SystemExit(f'checkpoint max_depth={checkpoint_max}, but --max-depth={args.max_depth}')

    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(_strip_module(state_dict), strict=True)
    test_start = time.perf_counter()
    test_metrics, test_frames = evaluate(
        model, test_loader, device, args, 'test',
        os.path.join(vis_root, 'test'), args.test_vis_num)
    best_epoch = (int(checkpoint.get('epoch', fallback_epoch))
                  if isinstance(checkpoint, dict) else fallback_epoch)
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


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser('Train/validate/test fisheye metric DA-V2')
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
    parser.add_argument('--sky-as-far', action='store_true')
    # Model / optimizer
    parser.add_argument('--encoder', default='vitb', choices=MODEL_CONFIGS)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--min-depth', type=float, default=0.2)
    parser.add_argument('--metric-depth-max', type=float, default=15.0,
                        help='largest distance trained with exact metric SiLog loss')
    parser.add_argument('--far-depth', type=float, default=19.0,
                        help='minimum acceptable prediction on far-mask pixels')
    parser.add_argument('--max-depth', type=float, default=20.0)
    parser.add_argument('--far-loss-weight', type=float, default=0.1,
                        help='weight of one-sided far loss relative to SiLog')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--pretrained-from', default=None)
    parser.add_argument('--load-head', action='store_true')
    parser.add_argument('--resume', default=None,
                        help='resume from last.pt (optimizer/scaler state required)')
    parser.add_argument('--model-training', type=parse_bool, default=True,
                        help='true: train/validate/test; false: test inference only')
    parser.add_argument('--checkpoint', default=None,
                        help='best.pt used when --model-training false')
    # Output / visualization
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--verbose-vis', action='store_true',
                        help='save validation figures at regular epoch intervals')
    parser.add_argument('--vis-epoch-interval', '--vis-every', dest='vis_epoch_interval',
                        type=int, default=5)
    parser.add_argument('--valid-vis-num', '--num-vis', dest='valid_vis_num',
                        type=int, default=6, help='validation figures to save; 0 means all')
    parser.add_argument('--test-vis-num', type=int, default=0,
                        help='test figures to save after training; 0 means all')
    parser.add_argument('--dmin', type=float, default=0.2,
                        help='shared visualization/colorbar minimum only')
    parser.add_argument('--dmax', type=float, default=20.0,
                        help='shared visualization/colorbar maximum only')
    parser.add_argument('--no-tb', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.vis_epoch_interval <= 0:
        parser.error('--vis-epoch-interval must be positive')
    if not args.min_depth < args.metric_depth_max < args.far_depth < args.max_depth:
        parser.error('require min-depth < metric-depth-max < far-depth < max-depth')
    if args.far_loss_weight < 0:
        parser.error('--far-loss-weight must be non-negative')
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
          f'target={args.target_key}')

    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder],
                               'max_depth': args.max_depth})
    if args.pretrained_from:
        load_pretrained(model, args.pretrained_from, args.load_head)
    model = model.to(device)

    if not args.model_training:
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
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')

    start_epoch, best, best_epoch = 0, float('inf'), -1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise SystemExit(f'resume checkpoint not found: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
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
    metrics_csv = os.path.join(args.save_path, 'metrics.csv')
    best_path = os.path.join(args.save_path, 'best.pt')

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_total_loss = 0.0
        running_metric_loss = 0.0
        running_far_loss = 0.0
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
                metric_loss = (criterion(prediction, depth, mask)
                               if mask.sum() >= 10 else prediction.sum() * 0.0)
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
            running_far_loss += far_loss.item()
            batches += 1
            if writer and global_iteration % 20 == 0:
                writer.add_scalar('train/loss_total', loss.item(), global_iteration)
                writer.add_scalar('train/loss_metric', metric_loss.item(), global_iteration)
                writer.add_scalar('train/loss_far', far_loss.item(), global_iteration)
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
        train_far_loss = running_far_loss / max(batches, 1)
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
            'train_silog_loss': train_metric_loss,
            'train_far_loss': train_far_loss,
            **prefixed_metrics('val', val_metrics),
            'val_evaluated_frames': val_frames,
            'lr_encoder': optimizer.param_groups[0]['lr'],
            'lr_head': optimizer.param_groups[1]['lr'],
            'epoch_seconds': time.perf_counter() - epoch_start,
            'is_best': is_best,
        }
        append_csv(metrics_csv, row)
        print(f'[epoch {epoch:03d}] total={train_total_loss:.4f} '
              f'metric={train_metric_loss:.4f} far={train_far_loss:.4f} '
              f"val_abs_rel={val_metrics['abs_rel']:.4f} "
              f"val_mae={val_metrics['mae']:.3f}m "
              f"val_rmse={val_metrics['rmse']:.3f}m d1={val_metrics['d1']:.4f}")
        if writer:
            writer.add_scalar('train/loss_total_epoch', train_total_loss, epoch)
            writer.add_scalar('train/loss_metric_epoch', train_metric_loss, epoch)
            writer.add_scalar('train/loss_far_epoch', train_far_loss, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)

    if not os.path.isfile(best_path):
        raise SystemExit('no best.pt was produced; validation had no usable frames')

    run_test(model, best_path, test_loader, device, args, vis_root,
             checkpoint_label='best.pt', fallback_epoch=best_epoch, writer=writer)
    if writer:
        writer.close()
    print(f'done. outputs: {args.save_path}')


if __name__ == '__main__':
    main()
