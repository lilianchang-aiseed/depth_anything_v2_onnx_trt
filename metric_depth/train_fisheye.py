"""
Distill (Depth-Anything-V2-Small relative + per-frame RealSense affine) into a
single metric Depth-Anything-V2 network.

Target per pixel is `depth_aligned` from make_gt_depthanything.py. After training,
deploy is ONE monocular forward pass -> metric depth in metres, with no RealSense
and no SML at inference.

Single-GPU, non-distributed (run it on a desktop/server GPU, not the Orin).
Reuses the upstream model / transforms / SiLog loss / metrics unchanged.

Example:
    python train_fisheye.py \
        --data /path/to/export_dir \
        --encoder vitb --img-size 518 \
        --min-depth 0.2 --max-depth 20 \
        --bs 4 --epochs 60 --lr 5e-6 --amp \
        --pretrained-from ../checkpoints/depth_anything_v2_vitb.pth \
        --save-path exp/fisheye_vitb
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


# --------------------------------------------------------------------------- #
# Data splitting
# --------------------------------------------------------------------------- #
def chunked_split(files, chunk=20, val_stride=5):
    """Group temporally-adjacent frames into chunks; every val_stride-th chunk
    goes to val. Keeps near-duplicate consecutive frames out of both splits."""
    files = sorted(files)
    train, val = [], []
    for i in range(0, len(files), chunk):
        (val if (i // chunk) % val_stride == 0 else train).extend(files[i:i + chunk])
    return train, val


def write_list(paths, out_path):
    with open(out_path, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    return out_path


# --------------------------------------------------------------------------- #
# Checkpoint / pretrained
# --------------------------------------------------------------------------- #
def _strip_module(sd):
    return {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}


def load_pretrained(model, path, load_head):
    """Warm-start weights.

    load_head=False (default): load ONLY the DINOv2 encoder ('pretrained.*').
        Use with a RELATIVE checkpoint (depth_anything_v2_<enc>.pth); the metric
        head then trains from its init.
    load_head=True: load the full state dict (encoder + metric head), strict=False.
        Use with a METRIC checkpoint to warm-start the head too.
    """
    raw = torch.load(path, map_location='cpu')
    sd = raw.get('model', raw) if isinstance(raw, dict) else raw
    sd = _strip_module(sd)
    if not load_head:
        sd = {k: v for k, v in sd.items() if 'pretrained' in k}
    ret = model.load_state_dict(sd, strict=False)
    print(f"[pretrained] {os.path.basename(path)}  load_head={load_head}: "
          f"loaded {len(sd)} tensors  (missing={len(ret.missing_keys)} "
          f"unexpected={len(ret.unexpected_keys)})")


def save_ckpt(path, model, optimizer, epoch, best, args):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best': best,
        'encoder': args.encoder,
        'img_size': args.img_size,
        'min_depth': args.min_depth,
        'max_depth': args.max_depth,
        'target_key': args.target_key,
    }, path)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _colorize(depth, mask, dmin, dmax):
    d = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    m = mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask) else np.asarray(mask, bool)
    m = m & np.isfinite(d)
    norm = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~m] = 40
    return col


def save_vis(path, depth_gt, valid, pred, dmin, dmax):
    gt_col = _colorize(depth_gt, valid, dmin, dmax)
    pr_col = _colorize(pred, torch.ones_like(pred, dtype=torch.bool), dmin, dmax)
    for img, txt in [(gt_col, 'GT (depth_aligned)'), (pr_col, 'SML-free metric pred')]:
        cv2.putText(img, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.hstack([gt_col, pr_col]))


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, args, save_dir=None, num_vis=6):
    model.eval()
    keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
    agg = {k: 0.0 for k in keys}
    n = 0
    saved = 0
    for sample in loader:
        img = sample['image'].to(device).float()
        depth = sample['depth'].to(device)[0]
        valid = sample['valid_mask'].to(device)[0]
        pred = model(img)
        pred = F.interpolate(pred[:, None], depth.shape[-2:],
                             mode='bilinear', align_corners=True)[0, 0]
        m = (valid == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        if m.sum() < 10:
            continue
        r = eval_depth(pred[m], depth[m])
        for k in keys:
            agg[k] += r[k]
        n += 1
        if save_dir and saved < num_vis:
            save_vis(os.path.join(save_dir, f'val_{saved:02d}.png'),
                     depth, m, pred, args.dmin, args.dmax)
            saved += 1
    if n == 0:
        return {k: float('nan') for k in keys}
    return {k: agg[k] / n for k in keys}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser('Distill DA-Small+affine into a metric DA-V2')
    # data
    ap.add_argument('--data', default=None, help='dir of .npz (auto train/val split)')
    ap.add_argument('--train-list', default=None, help='explicit train split .txt')
    ap.add_argument('--val-list', default=None, help='explicit val split .txt')
    ap.add_argument('--split-chunk', type=int, default=20)
    ap.add_argument('--val-stride', type=int, default=5)
    ap.add_argument('--target-key', default='depth_aligned',
                    choices=['depth_aligned', 'rs_depth_L'])
    ap.add_argument('--sky-as-far', action='store_true',
                    help='weakly label sky as max_depth so the net learns "sky=far"')
    # model / optim
    ap.add_argument('--encoder', default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    ap.add_argument('--img-size', type=int, default=518)
    ap.add_argument('--min-depth', type=float, default=0.2)
    ap.add_argument('--max-depth', type=float, default=20.0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true', help='mixed precision (fp16)')
    ap.add_argument('--pretrained-from', default=None)
    ap.add_argument('--load-head', action='store_true',
                    help='also load the metric head (use with a METRIC checkpoint)')
    ap.add_argument('--resume', default=None)
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
        assert args.data, 'pass --data DIR or both --train-list/--val-list'
        files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
        if not files:
            raise SystemExit(f'no .npz in {args.data}')
        tr, va = chunked_split(files, args.split_chunk, args.val_stride)
        train_list = write_list(tr, os.path.join(args.save_path, 'train.txt'))
        val_list = write_list(va, os.path.join(args.save_path, 'val.txt'))
        print(f'[split] {len(files)} files -> train={len(tr)} val={len(va)} '
              f'(chunk={args.split_chunk}, val_stride={args.val_stride})')

    size = (args.img_size, args.img_size)
    ds_kw = dict(size=size, target_key=args.target_key,
                 min_depth=args.min_depth, max_depth=args.max_depth,
                 sky_as_far=args.sky_as_far)
    train_ds = FisheyeNPZ(train_list, 'train', **ds_kw)
    val_ds = FisheyeNPZ(val_list, 'val', **ds_kw)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f'[data] train={len(train_ds)} val={len(val_ds)}  device={device}  '
          f'encoder={args.encoder}  img={args.img_size}  target={args.target_key}')

    # ---- model ----
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth})
    if args.pretrained_from:
        load_pretrained(model, args.pretrained_from, args.load_head)
    model = model.to(device)

    # two LR groups: encoder at lr, head at lr*10 (upstream schedule)
    optimizer = AdamW(
        [{'params': [p for n, p in model.named_parameters() if 'pretrained' in n], 'lr': args.lr},
         {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n], 'lr': args.lr * 10.0}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    criterion = SiLogLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')

    start_epoch, best = 0, 1e9
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(_strip_module(ck['model']), strict=False)
        try:
            optimizer.load_state_dict(ck['optimizer'])
        except Exception as e:
            print(f'[resume] optimizer state skipped ({e})')
        start_epoch = int(ck.get('epoch', -1)) + 1
        best = float(ck.get('best', 1e9))
        print(f'[resume] from epoch {start_epoch}, best abs_rel={best:.4f}')

    writer = SummaryWriter(args.save_path) if (_HAS_TB and not args.no_tb) else None
    total_iters = args.epochs * max(len(train_dl), 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, nb = 0.0, 0
        for i, sample in enumerate(train_dl):
            img = sample['image'].to(device)
            depth = sample['depth'].to(device)
            valid = sample['valid_mask'].to(device)

            if random.random() < 0.5:                    # horizontal flip
                img = img.flip(-1); depth = depth.flip(-1); valid = valid.flip(-1)

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
                print(f'  ep{epoch:03d} it{i:04d}/{len(train_dl)} '
                      f'lr={lr:.2e} loss={loss.item():.4f}')

        do_vis = (epoch % args.vis_every == 0)
        metrics = evaluate(model, val_dl, device, args,
                           save_dir=(vis_dir if do_vis else None), num_vis=args.num_vis)
        tr_loss = running / max(nb, 1)
        print(f'[ep {epoch:03d}] train_loss={tr_loss:.4f}  '
              f"abs_rel={metrics['abs_rel']:.4f}  rmse={metrics['rmse']:.4f}  "
              f"d1={metrics['d1']:.4f}")
        if writer:
            writer.add_scalar('train/loss_epoch', tr_loss, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

        save_ckpt(os.path.join(args.save_path, 'last.pt'), model, optimizer, epoch, best, args)
        if np.isfinite(metrics['abs_rel']) and metrics['abs_rel'] < best:
            best = metrics['abs_rel']
            save_ckpt(os.path.join(args.save_path, 'best.pt'), model, optimizer, epoch, best, args)
            print(f'  -> new best abs_rel={best:.4f}')

    if writer:
        writer.close()
    print(f'done. best abs_rel={best:.4f}. checkpoints in {args.save_path}')
    print('deploy: run best.pt monocularly -> metric depth (m). No RealSense, no SML.')


if __name__ == '__main__':
    main()