#!/usr/bin/env python3
"""Convert recorded stereo disparity in NPZ files to metric-depth NPY files.

The conversion matches ``stereo/ray_origin`` for pair 1_0:
320x320 disparity is 5x5 min-pooled to 64x64, then the per-pixel and global
inverse-depth scale/shift are applied. Existing 64x64 disparity is not pooled
again. Output is float32 metres with invalid pixels set to zero.

Example:
  python3 Depth-Anything-V2/ml_gt_generate/tools/stereo_depth_from_npz.py \
    --data Depth-Anything-V2/ml_gt_generate/dataset/train_1_0-5/train_1_0-5_supp_2 \
    --out-dir Depth-Anything-V2/ml_gt_generate/dataset/train_1_0-5/train_1_0-5_stereo_depth_2
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


ML_GT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ML_GT_ROOT.parent.parent
DEFAULT_CALIB = WORKSPACE_ROOT / "stereo/ray_origin/ros-stereo/rectify"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--pair", default="1_0")
    ap.add_argument("--disp-key", default="disp")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    masks = args.calib_dir / "masks"
    valid_mask = np.load(masks / f"pair_{args.pair}.npy").astype(bool)
    pixel_scale = np.load(masks / f"scale_{args.pair}.npy").astype(np.float32)
    pixel_shift = np.load(masks / f"shift_{args.pair}.npy").astype(np.float32)
    with (args.calib_dir / "stereo_rectified.yaml").open() as stream:
        params = yaml.safe_load(stream)[args.pair]
    global_scale = float(params["ransac_scale"])
    global_shift = float(params["ransac_shift"])

    files = sorted(args.data.glob("*.npz"))
    if not files:
        raise SystemExit(f"No NPZ files in {args.data}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    for index, path in enumerate(files, 1):
        output = args.out_dir / f"{path.stem}.npy"
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            if args.disp_key not in data.files:
                print(f"Skip {path.name}: missing {args.disp_key!r}")
                skipped += 1
                continue
            disparity = np.asarray(data[args.disp_key], dtype=np.float32).squeeze()

        if disparity.shape == (320, 320):
            disparity = disparity.reshape(64, 5, 64, 5).min(axis=(1, 3))
        elif disparity.shape != (64, 64):
            print(f"Skip {path.name}: unsupported disparity shape {disparity.shape}")
            skipped += 1
            continue

        corrected = disparity * pixel_scale + pixel_shift
        inverse_depth = corrected * global_scale + global_shift
        valid = (
            valid_mask
            & np.isfinite(disparity)
            & np.isfinite(inverse_depth)
            & (corrected > 0.1)
            & (inverse_depth > 0.05)
            & (inverse_depth < 10.0)
        )
        depth = np.zeros((64, 64), dtype=np.float32)
        np.divide(1.0, inverse_depth, out=depth, where=valid)
        np.save(output, depth)
        written += 1
        if index == 1 or index % 200 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {output}")

    print(f"Wrote {written}, skipped {skipped}: {args.out_dir}")


if __name__ == "__main__":
    main()
