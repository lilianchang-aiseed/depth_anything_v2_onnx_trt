"""
Whole-dataset evaluation + visualization for a trained metric DA-V2 checkpoint.

For every .npz it writes a 4-panel montage  [ left | GT | pred | abs-rel error ],
a per-frame metrics row to metrics.csv, and (optionally) stitches the montages
into an mp4 so you can scrub temporal consistency across a video. Prints the
aggregate metrics at the end.

    python eval_fisheye.py --ckpt exp/fisheye_vitb/best.pt \
        --data export_vid1 export_vid2 export_vid3 \
        --out eval_out --video --max-vis 0
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset.fisheye_npz import FisheyeNPZ
from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
KEYS = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']


def colorize(depth, mask, dmin, dmax, cmap=cv2.COLORMAP_TURBO):
    d = np.asarray(depth, np.float32)
    m = np.asarray(mask, bool) & np.isfinite(d)
    norm = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    col[~m] = 40
    return col


def error_panel(pred, gt, mask, emax=0.5):
    err = np.zeros_like(gt, np.float32)
    m = np.asarray(mask, bool) & np.isfinite(gt) & (gt > 0)
    err[m] = np.abs(pred[m] - gt[m]) / gt[m]
    col = cv2.applyColorMap((np.clip(err / emax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[~m] = 40
    return col


def label(img, txt):
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', nargs='+', default=None, help='dirs of .npz')
    ap.add_argument('--list', default=None, help='explicit .txt of .npz paths')
    ap.add_argument('--out', default='eval_out')
    ap.add_argument('--encoder', default=None)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--max-depth', type=float, default=None)
    ap.add_argument('--min-depth', type=float, default=0.2)
    ap.add_argument('--dmin', type=float, default=0.2)
    ap.add_argument('--dmax', type=float, default=20.0)
    ap.add_argument('--emax', type=float, default=0.5, help='error colormap ceiling (abs-rel)')
    ap.add_argument('--max-vis', type=int, default=0, help='cap montages written (0=all)')
    ap.add_argument('--video', action='store_true', help='also write montage.mp4')
    ap.add_argument('--fps', type=float, default=10.0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    encoder = args.encoder or ck.get('encoder', 'vitb')
    img_size = args.img_size or ck.get('img_size', 518)
    max_depth = args.max_depth or ck.get('max_depth', 20.0)

    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], 'max_depth': max_depth})
    ret = model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    print(f'[eval] {encoder} img={img_size} max_depth={max_depth} '
          f'(missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)})')

    # file list
    if args.list:
        list_path = args.list
    else:
        assert args.data, 'pass --data DIR... or --list FILE'
        files = [f for d in args.data for f in sorted(glob.glob(os.path.join(d, '*.npz')))]
        os.makedirs(args.out, exist_ok=True)
        list_path = os.path.join(args.out, '_all.txt')
        open(list_path, 'w').write('\n'.join(files) + '\n')

    os.makedirs(args.out, exist_ok=True)
    mont_dir = os.path.join(args.out, 'montage'); os.makedirs(mont_dir, exist_ok=True)
    ds = FisheyeNPZ(list_path, 'val', size=(img_size, img_size),
                    min_depth=args.min_depth, max_depth=max_depth, augment=False)

    agg = {k: 0.0 for k in KEYS}
    n = 0
    vw = None
    csv_f = open(os.path.join(args.out, 'metrics.csv'), 'w', newline='')
    writer = csv.writer(csv_f)
    writer.writerow(['file', 'n_valid'] + KEYS)

    for idx in range(len(ds)):
        sample = ds[idx]
        path = sample['image_path']
        img = sample['image'].unsqueeze(0).to(device).float()
        depth = sample['depth'].to(device)
        valid = sample['valid_mask'].to(device)
        with torch.no_grad():
            pred = model(img)
            pred = F.interpolate(pred[:, None], depth.shape[-2:],
                                 mode='bilinear', align_corners=True)[0, 0]
        m = (valid == 1) & (depth >= args.min_depth) & (depth <= max_depth)
        if int(m.sum()) < 10:
            continue
        r = eval_depth(pred[m], depth[m])
        for k in KEYS:
            agg[k] += r[k]
        n += 1
        writer.writerow([os.path.basename(path), int(m.sum())] +
                        [f'{r[k]:.5f}' for k in KEYS])

        if args.max_vis == 0 or n <= args.max_vis:
            z = np.load(path, allow_pickle=False)
            left = z['left']
            left_bgr = left[..., :3].astype(np.uint8) if left.ndim == 3 else \
                cv2.cvtColor(left.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_np = depth.cpu().numpy(); pr_np = pred.cpu().numpy(); m_np = m.cpu().numpy()
            panels = [
                label(cv2.resize(left_bgr, gt_np.shape[::-1]), 'left'),
                label(colorize(gt_np, m_np, args.dmin, args.dmax), 'GT (depth_aligned)'),
                label(colorize(pr_np, np.ones_like(m_np), args.dmin, args.dmax), 'metric pred'),
                label(error_panel(pr_np, gt_np, m_np, args.emax),
                      f'abs-rel err (<{args.emax:.1f})  frame={r["abs_rel"]:.3f}'),
            ]
            montage = np.hstack(panels)
            parent = os.path.basename(os.path.dirname(path))
            stem = f"{parent}__{os.path.splitext(os.path.basename(path))[0]}"
            cv2.imwrite(os.path.join(mont_dir, f'{stem}.png'), montage)
            if args.video:
                if vw is None:
                    h, w = montage.shape[:2]
                    vw = cv2.VideoWriter(os.path.join(args.out, 'montage.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
                vw.write(montage)
        if n % 100 == 0:
            print(f'  ...{n} frames  running abs_rel={agg["abs_rel"]/n:.4f}')

    csv_f.close()
    if vw is not None:
        vw.release()
    if n == 0:
        print('no valid frames'); return
    means = {k: agg[k] / n for k in KEYS}
    print(f'\n=== aggregate over {n} frames (mean-over-frames) ===')
    for k in KEYS:
        print(f'  {k:9s} {means[k]:.4f}')
    print(f'\nmontages -> {mont_dir}   csv -> {os.path.join(args.out,"metrics.csv")}'
          + (f'   video -> {os.path.join(args.out,"montage.mp4")}' if args.video else ''))


if __name__ == '__main__':
    main()