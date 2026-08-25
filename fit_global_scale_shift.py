#!/usr/bin/env python3
"""
Fit a constant (scale, shift) for deploy-time global alignment.

Reads every .npz training sample produced by the part-1 GT pipeline
(make_gt_depthanything.py) and fits

    ga_inv = scale * disp + shift    (units: 1/m)

against the RealSense-aligned GT depth at every valid pixel. Reports three
estimators of the "best single (scale, shift)" for a stereo pair:

  1. Per-frame lstsq  -> take median across frames.
  2. Pooled lstsq     -> one solve on all valid pixels concatenated.
  3. Pooled RANSAC    -> pooled but robust; rejects outlier pixels.

Also compares against the ransac_scale / ransac_shift currently stored in
stereo_rectified.yaml (if given) so you can see whether reality moved.

Usage:
    python fit_global_scale_shift.py --data ./merged_dataset --pair 1_0

    # write a YAML snippet you can paste into stereo_rectified.yaml
    python fit_global_scale_shift.py --data ./merged_dataset --pair 1_0 \\
        --out fitted_scale_shift.yaml

    # compare against your current deploy constants
    python fit_global_scale_shift.py --data ./merged_dataset --pair 1_0 \\
        --calib /home/nvidia/ros_stereo/rectify/stereo_rectified.yaml
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2                                       # only used for disp resize
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def per_frame_lstsq(disp: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Fit ga_inv = s*disp + t at ALL valid pixels of one frame (no subsampling)."""
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    d      = disp[ys, xs].astype(np.float64)
    inv_gt = 1.0 / gt[ys, xs].astype(np.float64)
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, inv_gt, rcond=None)
    return float(sol[0]), float(sol[1])


def pooled_lstsq(disp_all: np.ndarray, inv_gt_all: np.ndarray):
    """One lstsq solve over the concatenated anchor pool."""
    A = np.stack([disp_all, np.ones_like(disp_all)], axis=1)
    sol, *_ = np.linalg.lstsq(A, inv_gt_all, rcond=None)
    s, t = float(sol[0]), float(sol[1])
    pred = s * disp_all + t
    rmse = float(np.sqrt(np.mean((pred - inv_gt_all) ** 2)))
    return s, t, rmse


def pooled_ransac(disp_all: np.ndarray, inv_gt_all: np.ndarray,
                  n_iters: int = 1000, thresh: float = 0.02, seed: int = 0):
    """RANSAC on the pooled anchors. Returns (s, t, n_inliers, n_total) or None.

    Each iteration samples two random pairs, solves exactly, counts inliers
    at inv-depth L1 error < thresh (units: 1/m; 0.02 ~ 5% at 4m depth). Final
    (s, t) is the lstsq refit over the best inlier set.
    """
    n = len(disp_all)
    if n < 50:
        return None
    rng = np.random.default_rng(seed)
    best_inl_mask, best_count = None, 0
    for _ in range(n_iters):
        i1, i2 = rng.choice(n, 2, replace=False)
        d1, d2 = disp_all[i1], disp_all[i2]
        if abs(d1 - d2) < 1e-6:
            continue
        s = (inv_gt_all[i1] - inv_gt_all[i2]) / (d1 - d2)
        t = inv_gt_all[i1] - s * d1
        err = np.abs(s * disp_all + t - inv_gt_all)
        inl = err < thresh
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inl_mask = c, inl
    if best_inl_mask is None or best_count < 100:
        return None
    d = disp_all[best_inl_mask]
    y = inv_gt_all[best_inl_mask]
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1]), best_count, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='Directory of .npz samples from make_gt_depthanything.py')
    ap.add_argument('--pair', default='1_0',
                    help='Pair label written into --out YAML (does not affect math)')
    ap.add_argument('--out', default=None,
                    help='Optional YAML file to write the recommended values into')
    ap.add_argument('--calib', default=None,
                    help='Optional stereo_rectified.yaml to compare against')
    ap.add_argument('--max-samples', type=int, default=0,
                    help='0 = all files')
    ap.add_argument('--ransac-thresh', type=float, default=0.02,
                    help='RANSAC inlier threshold in 1/m (default 0.02 ~ 5%% at 4m)')
    ap.add_argument('--ransac-iters', type=int, default=1000)
    ap.add_argument('--ransac-pool-cap', type=int, default=200_000,
                    help='Cap on pool size for RANSAC (each iter is O(n))')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    if not files:
        raise SystemExit(f'no .npz files in {args.data}')
    if args.max_samples:
        files = files[:args.max_samples]
    print(f'Scanning {len(files)} files...')

    per_frame_st = []      # list of (s, t)
    all_disp     = []      # list of 1D arrays per frame
    all_inv_gt   = []
    n_used = n_skipped = n_bad = 0

    for i, path in enumerate(files):
        try:
            z = np.load(path, allow_pickle=False)
            if not bool(z['has_disp']):
                n_skipped += 1
                continue
            disp = z['disp'].astype(np.float32)
            gt   = z['depth_aligned'].astype(np.float32)
            H, W = gt.shape
            if disp.shape != (H, W):
                if cv2 is None:
                    raise RuntimeError('opencv required to resize disparity')
                disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_NEAREST)

            mask = (np.isfinite(gt) & (gt > 0) &
                    np.isfinite(disp) & (disp != 0))
            if mask.sum() < 20:
                n_skipped += 1
                continue

            st = per_frame_lstsq(disp, gt, mask)
            if st is None:
                n_skipped += 1
                continue
            per_frame_st.append(st)

            ys, xs = np.where(mask)
            all_disp.append(disp[ys, xs].astype(np.float64))
            all_inv_gt.append(1.0 / gt[ys, xs].astype(np.float64))
            n_used += 1
        except Exception as e:
            n_bad += 1
            if n_bad <= 5:
                print(f'  skipping {os.path.basename(path)}: {e}')
        if (i + 1) % 200 == 0:
            print(f'  ...{i+1}/{len(files)}  (used {n_used})')

    if not per_frame_st:
        raise SystemExit('no valid frames — nothing to fit')

    per_frame_st = np.asarray(per_frame_st)
    disp_pool    = np.concatenate(all_disp)
    inv_gt_pool  = np.concatenate(all_inv_gt)
    print(f'\nUsed {n_used} frames, skipped {n_skipped}, errored {n_bad}. '
          f'Pool size: {len(disp_pool):,} anchor pixels.\n')

    # ---- 1. per-frame lstsq -> median across frames ----------------------
    s_vals, t_vals = per_frame_st[:, 0], per_frame_st[:, 1]
    med_s, med_t = float(np.median(s_vals)), float(np.median(t_vals))
    print('--- per-frame lstsq (one fit per frame, then median) ---')
    print(f'  s: median={med_s:.6f}  mean={s_vals.mean():.6f}  '
          f'std={s_vals.std():.6f}  '
          f'p05..p95=[{np.percentile(s_vals,5):.6f}, {np.percentile(s_vals,95):.6f}]')
    print(f'  t: median={med_t:.6f}  mean={t_vals.mean():.6f}  '
          f'std={t_vals.std():.6f}  '
          f'p05..p95=[{np.percentile(t_vals,5):.6f}, {np.percentile(t_vals,95):.6f}]')

    # ---- 2. pooled lstsq -------------------------------------------------
    pool_s, pool_t, pool_rmse = pooled_lstsq(disp_pool, inv_gt_pool)
    print('\n--- pooled lstsq (all valid pixels equal weight, single solve) ---')
    print(f'  s={pool_s:.6f}  t={pool_t:.6f}  RMSE(1/m)={pool_rmse:.5f}')

    # ---- 3. pooled RANSAC ------------------------------------------------
    if len(disp_pool) > args.ransac_pool_cap:
        rng = np.random.default_rng(0)
        sub_idx = rng.choice(len(disp_pool), args.ransac_pool_cap, replace=False)
        d_sub, g_sub = disp_pool[sub_idx], inv_gt_pool[sub_idx]
        print(f'\n--- pooled RANSAC (subsampled to {args.ransac_pool_cap:,} '
              f'from {len(disp_pool):,}, thresh={args.ransac_thresh} 1/m, '
              f'iters={args.ransac_iters}) ---')
    else:
        d_sub, g_sub = disp_pool, inv_gt_pool
        print(f'\n--- pooled RANSAC (thresh={args.ransac_thresh} 1/m, '
              f'iters={args.ransac_iters}) ---')
    r = pooled_ransac(d_sub, g_sub, args.ransac_iters, args.ransac_thresh)
    if r is None:
        print('  RANSAC failed to converge; falling back to pooled lstsq')
        ran_s, ran_t = pool_s, pool_t
        ran_note = '(RANSAC fell back to pooled lstsq)'
    else:
        ran_s, ran_t, n_inl, n_tot = r
        print(f'  s={ran_s:.6f}  t={ran_t:.6f}  '
              f'inliers={n_inl:,}/{n_tot:,} ({100*n_inl/n_tot:.1f}%)')
        ran_note = f'({n_inl:,}/{n_tot:,} inliers)'

    # ---- Compare against existing calibration if provided ----------------
    if args.calib:
        try:
            import yaml
            with open(args.calib) as f:
                calib = yaml.safe_load(f)
            if args.pair in calib:
                cur_s = float(calib[args.pair]['ransac_scale'])
                cur_t = float(calib[args.pair]['ransac_shift'])
                print(f'\n--- current stereo_rectified.yaml for pair {args.pair} ---')
                print(f'  s={cur_s:.6f}  t={cur_t:.6f}')
                for name, ns, nt in (('per-frame median', med_s, med_t),
                                     ('pooled lstsq',    pool_s, pool_t),
                                     ('pooled RANSAC',   ran_s, ran_t)):
                    ds = 100 * (ns - cur_s) / abs(cur_s) if cur_s else float('nan')
                    dt = 100 * (nt - cur_t) / abs(cur_t) if cur_t else float('nan')
                    print(f'  vs {name:16s}: s {ds:+.1f}%   t {dt:+.1f}%')
            else:
                print(f'\n  (--calib provided but pair {args.pair} not in it)')
        except Exception as e:
            print(f'\n  (--calib load failed: {e})')

    # ---- Sanity: what depth distribution does each fit predict? ----------
    print('\n--- predicted depth (metres) using each fit vs GT ---')
    true_depth = 1.0 / inv_gt_pool
    print(f'  GT:                  p05={np.percentile(true_depth,5):5.2f}  '
          f'med={np.median(true_depth):5.2f}  '
          f'p95={np.percentile(true_depth,95):5.2f}')
    for name, s, t in (('per-frame median', med_s, med_t),
                       ('pooled lstsq',    pool_s, pool_t),
                       ('pooled RANSAC',   ran_s, ran_t)):
        pred_inv = np.clip(s * disp_pool + t, 1e-4, None)
        pred_d   = 1.0 / pred_inv
        print(f'  {name:16s}   p05={np.percentile(pred_d,5):5.2f}  '
              f'med={np.median(pred_d):5.2f}  '
              f'p95={np.percentile(pred_d,95):5.2f}')

    # ---- Recommendation --------------------------------------------------
    print('\n=== recommended (pooled RANSAC) — paste into stereo_rectified.yaml ===')
    print(f'{args.pair}:')
    print(f'  ransac_scale: {ran_s:.6f}   # {ran_note}')
    print(f'  ransac_shift: {ran_t:.6f}')
    print(f'  # alternative (per-frame median, more robust to outlier frames):')
    print(f'  # ransac_scale: {med_s:.6f}')
    print(f'  # ransac_shift: {med_t:.6f}')

    if args.out:
        try:
            import yaml
        except ImportError:
            raise SystemExit('  pip install pyyaml to write --out')
        payload = {args.pair: {'ransac_scale': ran_s, 'ransac_shift': ran_t,
                               'source': 'fit_global_scale_shift.py',
                               'note':   f'pooled RANSAC over {n_used} frames'}}
        with open(args.out, 'w') as f:
            yaml.safe_dump(payload, f, default_flow_style=False)
        print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()