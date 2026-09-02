#!/usr/bin/env python3
"""Compare D455, D435 and aligned DA-V2 depth in stereo-left coordinates.

The 2x4 montage contains:
  Left RGB | D455 depth overlay | D435 depth overlay | DA-V2 depth overlay
  D455+D435 | D455+DA-V2 | D435+DA-V2 | all three sources

Raw RealSense depth is streamed from the MCAP and projected to the 320x320
stereo-left grid.  Only one ROS message per source is retained in memory.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ML_GT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_GT_ROOT))

from make_gt.make_gt_depthanything import (  # noqa: E402
    CAM1_DIST,
    CAM1_PROJ,
    image_to_numpy,
    stamp_to_sec,
    step2_ir_depth_to_L,
)


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
D455_DEPTH = "/d455/d455_node/depth/image_rect_raw"
D435_DEPTH = "/d435/d435_node/depth/image_rect_raw"
D455_INFO = "/d455/d455_node/depth/camera_info"
D435_INFO = "/d435/d435_node/depth/camera_info"


def load_yaml(path):
    text = Path(path).read_text(encoding="utf-8").replace("%YAML:1.0", "")
    return yaml.safe_load(text)


def build_transforms(root):
    calib = root / "calib/rect_4cam_imu_in-extinsic"
    left_cfg = load_yaml(
        calib / "cam2imu/flight_data_2026_08_04-18_11_04_1-camchain-imucam.yaml"
    )
    d455_cfg = load_yaml(
        calib / "cam2realsense2imu/rs_d455_cube-imu_calib/"
        "flight_data_2026_08_27-10_27_07_0-camchain-imucam.yaml"
    )
    pair_cfg = load_yaml(
        root / "calib/rs_2cam_in-extrinsic/"
        "flight_data_2026_08_27-10_27_07_0-camchain.yaml"
    )
    t_left_imu = np.asarray(left_cfg["cam0"]["T_cam_imu"], np.float64)
    t_i455_imu = np.asarray(d455_cfg["cam0"]["T_cam_imu"], np.float64)
    t_i435_i455 = np.asarray(pair_cfg["cam1"]["T_cn_cnm1"], np.float64)
    t_left_i455 = t_left_imu @ np.linalg.inv(t_i455_imu)
    t_left_i435 = t_left_i455 @ np.linalg.inv(t_i435_i455)
    return t_left_i455, t_left_i435


def read_intrinsics(bag):
    wanted = {D455_INFO, D435_INFO}
    result = {}
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            k = np.asarray(msg.k, np.float64).reshape(3, 3)
            result[conn.topic] = (k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            if len(result) == len(wanted):
                break
    missing = wanted.difference(result)
    if missing:
        raise RuntimeError(f"Missing CameraInfo topics: {sorted(missing)}")
    return result[D455_INFO], result[D435_INFO]


def depth_color(depth, dmin, dmax):
    valid = np.isfinite(depth) & (depth >= dmin) & (depth <= dmax)
    norm = np.zeros(depth.shape, np.uint8)
    norm[valid] = np.clip(
        (depth[valid] - dmin) * 255.0 / max(dmax - dmin, 1e-6), 0, 255
    ).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return color, valid


def blend(base, color, valid, alpha):
    out = base.copy()
    out[valid] = cv2.addWeighted(base[valid], 1.0 - alpha, color[valid], alpha, 0)
    return out


def source_overlay(base, layers, alpha):
    out = base.astype(np.float32)
    for mask, bgr in layers:
        valid = mask.astype(bool)
        color = np.asarray(bgr, np.float32)
        out[valid] = (1.0 - alpha) * out[valid] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def tile(image, title, subtitle=""):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 38), (0, 0, 0), -1)
    cv2.putText(out, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return out


def make_montage(sample, d455, d435, dt455, dt435, alpha, dmin, dmax):
    left = sample["left"]
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    da = np.asarray(sample["depth_aligned"], np.float32)
    c455, v455 = depth_color(d455, dmin, dmax)
    c435, v435 = depth_color(d435, dmin, dmax)
    cda, vda = depth_color(da, dmin, dmax)
    stamp = float(sample["stamp"])

    cyan, magenta, yellow = (255, 255, 0), (255, 0, 255), (0, 255, 255)
    row1 = [
        tile(left, "Stereo-left RGB", f"stamp={stamp:.6f}"),
        tile(blend(left, c455, v455, alpha), "D455 depth + RGB",
             f"alpha={alpha:.2f}, dt={dt455 * 1e3:+.1f} ms"),
        tile(blend(left, c435, v435, alpha), "D435 depth + RGB",
             f"alpha={alpha:.2f}, dt={dt435 * 1e3:+.1f} ms"),
        tile(blend(left, cda, vda, alpha), "Aligned DA-V2 + RGB",
             f"shared scale={dmin:g}-{dmax:g} m"),
    ]
    row2 = [
        tile(source_overlay(left, [(v455, cyan), (v435, magenta)], alpha),
             "D455 + D435", "cyan=D455, magenta=D435"),
        tile(source_overlay(left, [(v455, cyan), (vda, yellow)], alpha),
             "D455 + DA-V2", "cyan=D455, yellow=DA-V2"),
        tile(source_overlay(left, [(v435, magenta), (vda, yellow)], alpha),
             "D435 + DA-V2", "magenta=D435, yellow=DA-V2"),
        tile(source_overlay(left, [(v455, cyan), (v435, magenta), (vda, yellow)], alpha),
             "All sources", "fixed source colors; common left grid"),
    ]
    return np.vstack([np.hstack(row1), np.hstack(row2)])


def stream_project(bag, targets, stems, cache, intrinsics, transforms, tol):
    topics = {D455_DEPTH: "d455", D435_DEPTH: "d435"}
    states = {topic: {"index": 0, "prev": None} for topic in topics}
    dt = {"d455": np.full(len(targets), np.nan),
          "d435": np.full(len(targets), np.nan)}

    def save_match(topic, index, item):
        if item is None:
            return
        stamp, msg = item
        delta = stamp - targets[index]
        if abs(delta) > tol:
            return
        device = topics[topic]
        depth = image_to_numpy(msg).astype(np.float32) * 0.001
        projected = step2_ir_depth_to_L(
            depth, intrinsics[device], transforms[device], CAM1_PROJ, CAM1_DIST,
            (320, 320), splat=True,
        )
        np.save(cache / device / f"{stems[index]}.npy", projected)
        dt[device][index] = delta

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=conns):
            state = states[conn.topic]
            if state["index"] >= len(targets):
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            current = (stamp_to_sec(msg), msg)
            while state["index"] < len(targets) and targets[state["index"]] <= current[0]:
                previous = state["prev"]
                chosen = current if previous is None else min(
                    (previous, current), key=lambda item: abs(item[0] - targets[state["index"]])
                )
                save_match(conn.topic, state["index"], chosen)
                state["index"] += 1
            state["prev"] = current
            if all(s["index"] >= len(targets) for s in states.values()):
                break

    for topic, state in states.items():
        while state["index"] < len(targets):
            save_match(topic, state["index"], state["prev"])
            state["index"] += 1
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Mounted SMB 0827 directory")
    ap.add_argument("--bag", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=ML_GT_ROOT / "out_data/0827_depth_overlap_vis")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--sync-tol", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--depth-min", type=float, default=0.2)
    ap.add_argument("--depth-max", type=float, default=15.0)
    args = ap.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    bag = args.bag or (
        args.root / "train_1_0_20260827_164533/train_1_0_20260827_164533_0.mcap"
    )
    data_dir = args.data or (args.root / "train_1_0_164533_dataset")
    files = sorted(data_dir.glob("sample_*.npz"))[:args.max_files]
    if not files:
        raise SystemExit(f"No samples found in {data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "aligned_depth_cache"
    (cache / "d455").mkdir(parents=True, exist_ok=True)
    (cache / "d435").mkdir(parents=True, exist_ok=True)

    targets, stems = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            targets.append(float(sample["stamp"]))
        stems.append(path.stem)
    targets = np.asarray(targets, np.float64)

    k455, k435 = read_intrinsics(bag)
    t455, t435 = build_transforms(args.root)
    print(f"D455 K={tuple(round(x, 3) for x in k455)}")
    print(f"D435 K={tuple(round(x, 3) for x in k435)}")
    print(f"T_L<-I455 t={np.round(t455[:3, 3], 4)}")
    print(f"T_L<-I435 t={np.round(t435[:3, 3], 4)}")
    dt = stream_project(
        bag, targets, stems, cache,
        {"d455": k455, "d435": k435}, {"d455": t455, "d435": t435},
        args.sync_tol,
    )

    written = 0
    for i, path in enumerate(files):
        p455 = cache / "d455" / f"{path.stem}.npy"
        p435 = cache / "d435" / f"{path.stem}.npy"
        if not p455.is_file() or not p435.is_file():
            print(f"Skip {path.stem}: no synchronized RealSense pair")
            continue
        with np.load(path, allow_pickle=False) as sample:
            montage = make_montage(
                sample, np.load(p455), np.load(p435), dt["d455"][i], dt["d435"][i],
                args.alpha, args.depth_min, args.depth_max,
            )
        output = args.out_dir / f"{path.stem}.jpg"
        cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
        if written == 1 or written % 25 == 0:
            print(f"[{written}/{len(files)}] {output}", flush=True)
    print(f"Done: {written}/{len(files)} montages -> {args.out_dir}")


if __name__ == "__main__":
    main()
