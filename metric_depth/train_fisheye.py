"""
Distill (DA-V2-Small relative + per-frame RealSense affine) into a single metric
Depth-Anything-V2. Deploy = one monocular forward pass -> metric depth (m), no
RealSense, no SML. Single-GPU, non-distributed (run on a desktop GPU, not the Orin).

Splitting (data is a few continuous videos -> pass each video as its own --data dir):
  --split-mode per_video_temporal : within each video, tail --val-frac -> val,
                                     with --val-gap frames dropped at the boundary
                                     (low leakage; all scenes in both splits).
  --split-mode holdout_video      : --holdout-video K holds video K entirely out.
  --split-mode chunk              : legacy interleaved chunks (leaky on video; avoid).
  --val-data DIR...               : use these dirs as val outright (honest held-out).

Checkpoints: best_absrel.pt / best_rmse.pt / best_d1.pt saved on their own metric;
best.pt mirrors whichever --select-metric you choose (default abs_rel).
"""
import argparse
import glob
import os
import random

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
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
BEST_DIRECTION = {'abs_rel': 'min', 'rmse': 'min', 'd1': 'max'}


# --------------------------------------------------------------------------- #
# Splitting  (each --data dir == one video)
# --------------------------------------------------------------------------- #
def list_videos(data_dirs):
    videos = []
    for d in data_dirs:
        files = sorted(glob.glob(os.path.join(d, '*.npz')))
        if files:
            videos.append((os.path.basename(os.path.normpath(d)) or d, files))
        else:
            print(f'[split] WARNING: no .npz in {d}')
    if not videos:
        raise SystemExit('no .npz found in any --data dir')
    return videos


def split_per_video_temporal(videos, val_frac, gap):
    train, val = [], []
    for name, files in videos:
        n = len(files)
        n_val = max(1, int(round(n * val_frac)))
        val_start = n - n_val
        tr_end = max(0, val_start - gap)
        train += files[:tr_end]
        val += files[val_start:]
        print(f'  [{name}] n={n} -> train={tr_end} gap={max(0,val_start-tr_end)} val={n_val}')
    return train, val


def split_holdout_video(videos, holdout_idx):
    train, val = [], []
    for i, (name, files) in enumerate(videos):
        (val if i == holdout_idx else train).extend(files)
        print(f'  [{name}] {"VAL (held out)" if i == holdout_idx else "train"}: {len(files)}')
    return train, val


def split_chunk(videos, chunk, val_stride):
    files = sorted(f for _, fs in videos for f in fs)
    train, val = [], []
    for i in range(0, len(files), chunk):
        (val if (i // chunk) % val_stride == 0 else train).extend(files[i:i + chunk])
    return train, val


def write_list(paths, out_path):
    with open(out_path, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    return out_path


# --------------------------------------------------------------------------- #
# Checkpoints / pretrained
# --------------------------------------------------------------------------- #
def _strip_module(sd):
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}


def load_pretrained(model, path, load_head):
    raw = torch.load(path, map_location='cpu')
    sd = raw.get('model', raw) if isinstance(raw, dict) else raw
    sd = _strip_module(sd)
    if not load_head:
        sd = {k: v for k, v in sd.items() if 'pretrained' in k}
    ret = model.load_state_dict(sd, strict=False)
    print(f"[pretrained] {os.path.basename(path)}  load_head={load_head}: "
          f"loaded {len(sd)} tensors (missing={len(ret.missing_keys)} "
          f"unexpected={len(ret.unexpected_keys)})")


def save_ckpt(path, model, optimizer, epoch, bests, args):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'bests': bests,
        'encoder': args.encoder,
        'img_size': args.img_size,
        'min_depth': args.min_depth,
        'max_depth': args.max_depth,
        'target_key': args.target_key,
    }, path)


# --------------------------------------------------------------------------- #
# Vis
# --------------------------------------------------------------------------- #
def _colorize(depth, mask, dmin, dmax):
    d = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    m = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    m = m.astype(bool) & np.isfinite(d)
    norm = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~m] = 40
    return col


def save_vis(path, depth_gt, valid, pred, dmin, dmax):
    gt_col = _colorize(depth_gt, valid, dmin, dmax)
    pr_col = _colorize(pred, torch.ones_like(pred, dtype=torch.bool), dmin, dmax)
    for img, txt in [(gt_col, 'GT (depth_aligned)'), (pr_col, 'metric pred')]:
        cv2.putText(img, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.hstack([gt_col, pr_col]))


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, args, save_dir=None, num_vis=6, criterion=None):
    model.eval()
    keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
    agg = {k: 0.0 for k in keys}
    loss_sum, loss_n = 0.0, 0
    n = saved = 0
    for sample in loader:
        img = sample['image'].to(device).float()
        depth_full = sample['depth'].to(device)                          # (1,H,W)
        valid_full = sample['valid_mask'].to(device)
        pred_raw = model(img)                                            # (1,h,w) at net res
        pred = F.interpolate(pred_raw[:, None], depth_full.shape[-2:],
                             mode='bilinear', align_corners=True)[0, 0]
        d = depth_full[0]; v = valid_full[0]
        m = (v == 1) & (d >= args.min_depth) & (d <= args.max_depth)
        if m.sum() < 10:
            continue
        # val SiLog loss at native (interpolated) res, same mask the metrics use
        if criterion is not None:
            l = criterion(pred.unsqueeze(0), d.unsqueeze(0), m.unsqueeze(0))
            if torch.isfinite(l):
                loss_sum += float(l.item()); loss_n += 1
        r = eval_depth(pred[m], d[m])
        for k in keys:
            agg[k] += r[k]
        n += 1
        if save_dir and saved < num_vis:
            save_vis(os.path.join(save_dir, f'val_{saved:02d}.png'),
                     d, m, pred, args.dmin, args.dmax)
            saved += 1
    out = {k: (agg[k] / n if n else float('nan')) for k in keys}
    out['loss'] = (loss_sum / loss_n) if loss_n else float('nan')
    return out


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser('Distill DA-Small+affine into metric DA-V2')
    # data / split
    ap.add_argument('--data', nargs='+', default=None,
                    help='one or more dirs of .npz; EACH DIR IS TREATED AS ONE VIDEO')
    ap.add_argument('--val-data', nargs='*', default=None,
                    help='dirs used as val outright (honest held-out set)')
    ap.add_argument('--train-list', default=None)
    ap.add_argument('--val-list', default=None)
    ap.add_argument('--split-mode', default='per_video_temporal',
                    choices=['per_video_temporal', 'holdout_video', 'chunk'])
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--val-gap', type=int, default=30,
                    help='frames dropped between train and val segments per video')
    ap.add_argument('--holdout-video', type=int, default=None,
                    help='index into sorted --data dirs to hold out entirely')
    ap.add_argument('--split-chunk', type=int, default=20)
    ap.add_argument('--val-stride', type=int, default=5)
    ap.add_argument('--target-key', default='depth_aligned',
                    choices=['depth_aligned', 'rs_depth_L'])
    ap.add_argument('--sky-as-far', action='store_true')
    # augmentation
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--no-hflip', action='store_true')
    ap.add_argument('--aug-brightness', type=float, default=0.3)
    ap.add_argument('--aug-contrast', type=float, default=0.2)
    ap.add_argument('--aug-gamma', type=float, default=0.0)
    ap.add_argument('--aug-noise-std', type=float, default=0.0118)
    ap.add_argument('--aug-noise-p', type=float, default=0.3)
    # model / optim
    ap.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    ap.add_argument('--img-size', type=int, default=518)
    ap.add_argument('--min-depth', type=float, default=0.2)
    ap.add_argument('--max-depth', type=float, default=20.0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--pretrained-from', default=None)
    ap.add_argument('--load-head', action='store_true')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--select-metric', default='abs_rel', choices=['abs_rel', 'rmse', 'd1'],
                    help='which metric best.pt mirrors')
    # io / vis
    ap.add_argument('--save-path', required=True)
    ap.add_argument('--vis-every', type=int, default=5)
    ap.add_argument('--num-vis', type=int, default=6)
    ap.add_argument('--dmin', type=float, default=0.2)
    ap.add_argument('--dmax', type=float, default=20.0)
    ap.add_argument('--no-tb', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_path, exist_ok=True)
    vis_dir = os.path.join(args.save_path, 'vis'); os.makedirs(vis_dir, exist_ok=True)

    # ---- resolve splits ----
    if args.train_list and args.val_list:
        train_list, val_list = args.train_list, args.val_list
    else:
        assert args.data, 'pass --data DIR... (or --train-list/--val-list)'
        videos = list_videos(args.data)
        if args.val_data:
            val_videos = list_videos(args.val_data)
            tr = [f for _, fs in videos for f in fs]
            va = [f for _, fs in val_videos for f in fs]
            print(f'[split] held-out val dirs: train={len(tr)} val={len(va)}')
        elif args.split_mode == 'holdout_video':
            assert args.holdout_video is not None, '--holdout-video required'
            print(f'[split] holdout_video={args.holdout_video}')
            tr, va = split_holdout_video(videos, args.holdout_video)
        elif args.split_mode == 'chunk':
            tr, va = split_chunk(videos, args.split_chunk, args.val_stride)
            print(f'[split] chunk (leaky on video): train={len(tr)} val={len(va)}')
        else:
            print(f'[split] per_video_temporal val_frac={args.val_frac} gap={args.val_gap}')
            tr, va = split_per_video_temporal(videos, args.val_frac, args.val_gap)
        train_list = write_list(tr, os.path.join(args.save_path, 'train.txt'))
        val_list = write_list(va, os.path.join(args.save_path, 'val.txt'))

    # ---- datasets ----
    size = (args.img_size, args.img_size)
    common = dict(size=size, target_key=args.target_key,
                  min_depth=args.min_depth, max_depth=args.max_depth,
                  sky_as_far=args.sky_as_far)
    train_ds = FisheyeNPZ(
        train_list, 'train', augment=not args.no_augment,
        aug_hflip=not args.no_hflip, aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast, aug_gamma=args.aug_gamma,
        aug_noise_std=args.aug_noise_std, aug_noise_p=args.aug_noise_p, **common)
    val_ds = FisheyeNPZ(val_list, 'val', augment=False, **common)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f'[data] train={len(train_ds)} val={len(val_ds)}  device={device}  '
          f'encoder={args.encoder}  img={args.img_size}  aug={not args.no_augment}')

    # ---- model / optim ----
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth})
    if args.pretrained_from:
        load_pretrained(model, args.pretrained_from, args.load_head)
    model = model.to(device)
    optimizer = AdamW(
        [{'params': [p for n, p in model.named_parameters() if 'pretrained' in n], 'lr': args.lr},
         {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n], 'lr': args.lr * 10.0}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    criterion = SiLogLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')

    bests = {'abs_rel': 1e9, 'rmse': 1e9, 'd1': -1.0}
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(_strip_module(ck['model']), strict=False)
        try:
            optimizer.load_state_dict(ck['optimizer'])
        except Exception as e:
            print(f'[resume] optimizer skipped ({e})')
        start_epoch = int(ck.get('epoch', -1)) + 1
        bests = ck.get('bests', bests)
        print(f'[resume] epoch {start_epoch}  bests={bests}')

    writer = SummaryWriter(args.save_path) if (_HAS_TB and not args.no_tb) else None
    total_iters = max(args.epochs, 1) * max(len(train_dl), 1)

    def _run_eval_and_save(epoch, do_vis):
        metrics = evaluate(model, val_dl, device, args,
                           save_dir=(vis_dir if do_vis else None),
                           num_vis=args.num_vis, criterion=criterion)
        for metric, direction in BEST_DIRECTION.items():
            v = metrics[metric]
            if not np.isfinite(v):
                continue
            improved = v < bests[metric] if direction == 'min' else v > bests[metric]
            if improved:
                bests[metric] = v
                save_ckpt(os.path.join(args.save_path, f'best_{metric}.pt'),
                          model, optimizer, epoch, bests, args)
                if metric == args.select_metric:
                    save_ckpt(os.path.join(args.save_path, 'best.pt'),
                              model, optimizer, epoch, bests, args)
                print(f'  -> new best {metric}={v:.4f}')
        return metrics

    if args.epochs == 0:   # eval-only
        m = _run_eval_and_save(start_epoch, do_vis=True)
        print('eval-only:', {k: round(m[k], 4) for k in ['abs_rel', 'rmse', 'd1']})
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, nb = 0.0, 0
        for i, sample in enumerate(train_dl):
            img = sample['image'].to(device)
            depth = sample['depth'].to(device)
            valid = sample['valid_mask'].to(device)
            mask = (valid == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            if mask.sum() < 10:
                continue
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp and device == 'cuda'):
                pred = model(img)
                loss = criterion(pred, depth, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            it = epoch * len(train_dl) + i
            lr = args.lr * (1 - it / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0
            running += loss.item(); nb += 1
            if writer and it % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), it)
                writer.add_scalar('train/lr', lr, it)
            if i % 50 == 0:
                print(f'  ep{epoch:03d} it{i:04d}/{len(train_dl)} lr={lr:.2e} loss={loss.item():.4f}')

        metrics = _run_eval_and_save(epoch, do_vis=(epoch % args.vis_every == 0))
        tr_loss = running / max(nb, 1)
        va_loss = metrics.get('loss', float('nan'))
        print(f'[ep {epoch:03d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  '
              f'abs_rel={metrics["abs_rel"]:.4f}  rmse={metrics["rmse"]:.4f}  d1={metrics["d1"]:.4f}')
        if writer:
            writer.add_scalar('train/loss_epoch', tr_loss, epoch)
            writer.add_scalar('val/loss', va_loss, epoch)
            # side-by-side train vs val on one tensorboard chart
            writer.add_scalars('loss', {'train': tr_loss, 'val': va_loss}, epoch)
            for k, v in metrics.items():
                if k == 'loss':
                    continue
                writer.add_scalar(f'val/{k}', v, epoch)
        save_ckpt(os.path.join(args.save_path, 'last.pt'), model, optimizer, epoch, bests, args)

    if writer:
        writer.close()
    print(f'done. bests={ {k: round(v,4) for k,v in bests.items()} }  in {args.save_path}')
    print('deploy: run best.pt monocularly -> metric depth (m). No RealSense, no SML.')


if __name__ == '__main__':
    main()