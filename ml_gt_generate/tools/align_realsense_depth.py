#!/usr/bin/env python3
"""Project recorded D455/D435 depth into stereo_1_0 left-image pixels.

Outputs one float32 metric-depth NPY per source and sample:
  <out-dir>/d455/sample_XXXX.npy
  <out-dir>/d435/sample_XXXX.npy

The transform chain uses the normalized calibration files owned by
``ml_gt_generate``:
  D455 depth -> stereo-left
  D435 depth -> D455 depth -> stereo-left
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


ML_GT_ROOT = Path(__file__).resolve().parents[1]
CALIB_ROOT = ML_GT_ROOT / "calib/2026-08-25"
LEFT_D455_CALIB = (
    CALIB_ROOT
    / "stereo_d455_d435_extrinsic/d455-left/stereo_1_0_d455_composed-camchain.yaml"
)
D455_D435_CALIB = (
    CALIB_ROOT
    / "stereo_d455_d435_extrinsic/d455-d435/d455_d435_depth_composed-camchain.yaml"
)
sys.path.insert(0, str(ML_GT_ROOT))

from make_gt.correct_matrix_direction import load_d455_d435, load_left_d455  # noqa: E402


def load_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = text.replace("%YAML:1.0", "", 1)
    return yaml.safe_load(text)


def camera_resolution(path, topic):
    calibration = load_yaml(path)
    for name in ("cam0", "cam1"):
        camera = calibration[name]
        if camera.get("rostopic") == topic:
            return tuple(camera["resolution"])
    raise ValueError(f"Camera topic {topic!r} not found in {path}")


def build_calibration():
    left_pair = load_left_d455(LEFT_D455_CALIB)
    depth_pair = load_d455_d435(D455_D435_CALIB)

    result = {}
    for device, camera, calibration_path, transform in (
        ("d455", left_pair.d455, LEFT_D455_CALIB,
         left_pair.T_left_d455),
        ("d435", depth_pair.d435, D455_D435_CALIB,
         left_pair.T_left_d455 @ depth_pair.T_d455_d435),
    ):
        result[device] = {
            "intrinsics": tuple(camera.intrinsics),
            "resolution": camera_resolution(calibration_path, camera.topic),
            "t_left_depth": np.asarray(transform, dtype=np.float64),
        }
    result["left"] = {
        "intrinsics": tuple(left_pair.left.intrinsics),
        "distortion": tuple(left_pair.left.distortion),
        "resolution": camera_resolution(LEFT_D455_CALIB, left_pair.left.topic),
    }
    return result


def project_depth(depth, source, target):
    height, width = depth.shape
    if (width, height) != source["resolution"]:
        raise ValueError(
            f"depth shape {depth.shape} does not match {source['resolution']}"
        )
    fx, fy, cx, cy = source["intrinsics"]
    # 65.535 m is uint16=65535 converted with the 0.001 m depth unit: a
    # saturated sensor code, not a measured range. Do not apply a general
    # maximum-range cutoff; only remove this exact invalid representation.
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.534)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    points = np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)), axis=1
    )
    points = (source["t_left_depth"] @ points.T).T[:, :3]
    z_left = points[:, 2]
    front = np.isfinite(points).all(axis=1) & (z_left > 0)
    points, z_left = points[front], z_left[front]

    x = points[:, 0] / z_left
    y = points[:, 1] / z_left
    k1, k2, p1, p2 = target["distortion"][:4]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    fx_t, fy_t, cx_t, cy_t = target["intrinsics"]
    uu = np.rint(fx_t * xd + cx_t).astype(np.int32)
    vv = np.rint(fy_t * yd + cy_t).astype(np.int32)
    out_width, out_height = target["resolution"]
    inside = (uu >= 0) & (uu < out_width) & (vv >= 0) & (vv < out_height)

    flat = np.full(out_width * out_height, np.inf, dtype=np.float32)
    indices = vv[inside] * out_width + uu[inside]
    np.minimum.at(flat, indices, z_left[inside].astype(np.float32))
    aligned = flat.reshape(out_height, out_width)
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(args.data.glob("*.npz"))
    if not files:
        raise SystemExit(f"No NPZ files in {args.data}")
    calib = build_calibration()
    for device in ("d455", "d435"):
        (args.out_dir / device).mkdir(parents=True, exist_ok=True)

    written = 0
    for index, path in enumerate(files, 1):
        outputs = {
            device: args.out_dir / device / f"{path.stem}.npy"
            for device in ("d455", "d435")
        }
        if not args.overwrite and all(path.is_file() for path in outputs.values()):
            continue
        with np.load(path, allow_pickle=False) as sample:
            for device, output in outputs.items():
                if output.exists() and not args.overwrite:
                    continue
                key = f"depth_{device}"
                if key not in sample.files:
                    raise KeyError(f"{path}: missing {key}")
                aligned = project_depth(sample[key], calib[device], calib["left"])
                np.save(output, aligned)
        written += 1
        if index == 1 or index % 200 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {path.stem}")
    print(f"Processed {written}/{len(files)} samples: {args.out_dir}")


if __name__ == "__main__":
    main()
