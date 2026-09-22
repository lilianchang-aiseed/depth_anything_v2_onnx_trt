#!/usr/bin/env python3
"""Compare D455/D435/LiDAR projections in one rectified-camera image plane.

Unlike ``depth_overlap_compare_0827.py``, this tool reads the target image and
raw sensor data directly from an MCAP.  It therefore needs no generated NPZ
dataset.  Available sources are selected automatically; an absent D455, D435,
or LiDAR is skipped rather than treated as an error.

The montage has one column per calibration candidate:

    target RGB | depth overlays for every candidate
    RGB edges  | projected depth edges for every candidate

``edge_distance_px`` is a diagnostic rather than ground truth: lower means
projected depth discontinuities lie closer to visible RGB edges. Pairwise depth
metrics compare target-camera Z in pixels where two projected maps overlap.

Usage:
source ~/.bashrc

d455
python3 ml_gt_generate/tools/depth_overlap_compare_bag.py \
  --bag "$COMMON_SHARE/bags/nx-2.0/0917/flight_data_2026_09_17-15_31_25/flight_data_2026_09_17-15_31_25_0.mcap" \
  --calib try1=calib/2026-09-16/try1/flight_data_2026_09_18-13_01_41_0-camchain.yaml \
  --calib try2=calib/2026-09-16/try2/flight_data_2026_09_18-13_10_31_0-camchain.yaml \
  --left-topic /camera_2/image_rect \
  --realsense d455 \
  --out-dir ml_gt_generate/out_data/0917_try1_try2_compare \
  --max-files 20 \
  --sync-tol 0.05

lidar+d455: 
python3 ml_gt_generate/tools/depth_overlap_compare_bag.py \
  --bag "$COMMON_SHARE/bags/nx-2.0/0916/flight_data_2026_09_16-15_23_09_newRect/flight_data_2026_09_16-15_23_09_newRect.mcap" \
  --left-topic /camera_2/image_rect \
  --sources d455,lidar \
  --calib d455=calib/2026-09-16/try1/flight_data_2026_09_18-13_01_41_0-camchain.yaml \
  --lidar-calib calib/2026-09-16/cam2-lidar-calib/lidar_calib_preprocess/calib.json \
  --out-dir ml_gt_generate/out_data/0916_camera2_d455_lidar_overlap \
  --max-files 20 \
  --sync-tol 0.05 \
  --depth-min 0.2 \
  --depth-max 15 \
  --alpha 0.5
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ML_GT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ML_GT_ROOT))

from make_gt.make_gt_depthanything import (  # noqa: E402
    image_to_numpy,
    stamp_to_sec,
    step2_ir_depth_to_L,
)
from tools.correct_matrix_direction import load_camera_lidar_json  # noqa: E402


TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
DEFAULT_DEPTH_TOPICS = {
    "d455": "/d455/d455_node/depth/image_rect_raw",
    "d435": "/d435/d435_node/depth/image_rect_raw",
}
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
LIDAR_VIS_RADIUS = 3  # Montage only; never used by metrics or coverage.

# Defaults are deliberately visible at the top of the script. CLI arguments
# may override them without requiring a source-code edit.
CALIBRATION_PATHS = {
    "camera_d455": (
        PROJECT_ROOT / "calib/2026-09-16/try1"
        / "flight_data_2026_09_18-13_01_41_0-camchain.yaml"
    ),
    "d455_d435": None,
    "camera_lidar": (
        PROJECT_ROOT / "calib/2026-09-16/cam2-lidar-calib"
        / "lidar_calib_preprocess/calib.json"
    ),
}


@dataclass(frozen=True)
class ProjectionCalibration:
    label: str
    path: Path
    source: str
    source_intrinsics: np.ndarray
    target_intrinsics: np.ndarray
    target_distortion: np.ndarray
    target_resolution: tuple
    T_target_source: np.ndarray


@dataclass(frozen=True)
class LidarProjectionCalibration:
    label: str
    path: Path
    source: str
    target_intrinsics: np.ndarray
    target_distortion: np.ndarray
    target_topic: str
    T_target_source: np.ndarray


def load_yaml(path):
    text = Path(path).read_text(encoding="utf-8").replace("%YAML:1.0", "")
    return yaml.safe_load(text)


def camera_source(topic):
    topic = str(topic).lower()
    if "d455" in topic:
        return "d455"
    if "d435" in topic:
        return "d435"
    return None


def parse_calibration(spec, left_topic):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--calib must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(
            f"invalid calibration {spec!r}; file exists={path.is_file()}")
    data = load_yaml(path)
    if not isinstance(data, dict) or "cam0" not in data or "cam1" not in data:
        raise argparse.ArgumentTypeError(f"{path}: expected cam0 and cam1")

    cameras = [data["cam0"], data["cam1"]]
    source_indices = [i for i, camera in enumerate(cameras)
                      if camera_source(camera.get("rostopic"))]
    if len(source_indices) != 1:
        raise argparse.ArgumentTypeError(
            f"{path}: expected exactly one D455/D435 camera")
    source_index = source_indices[0]
    target_index = 1 - source_index
    source = camera_source(cameras[source_index].get("rostopic"))
    target_topic = str(cameras[target_index].get("rostopic", ""))
    if left_topic and target_topic != left_topic:
        raise argparse.ArgumentTypeError(
            f"{path}: target topic is {target_topic!r}, expected {left_topic!r}")

    if "T_cn_cnm1" not in data["cam1"]:
        raise argparse.ArgumentTypeError(f"{path}: cam1.T_cn_cnm1 is missing")
    T_cam1_cam0 = np.asarray(data["cam1"]["T_cn_cnm1"], np.float64)
    if T_cam1_cam0.shape != (4, 4):
        raise argparse.ArgumentTypeError(f"{path}: transform is not 4x4")
    T_target_source = (T_cam1_cam0 if target_index == 1
                       else np.linalg.inv(T_cam1_cam0))
    source_camera = cameras[source_index]
    target_camera = cameras[target_index]
    resolution = tuple(map(int, target_camera.get("resolution", ())))
    if len(resolution) != 2:
        raise argparse.ArgumentTypeError(f"{path}: invalid target resolution")
    return ProjectionCalibration(
        label=label,
        path=path,
        source=source,
        source_intrinsics=np.asarray(source_camera["intrinsics"], np.float64),
        target_intrinsics=np.asarray(target_camera["intrinsics"], np.float64),
        target_distortion=np.asarray(
            target_camera.get("distortion_coeffs", []), np.float64),
        target_resolution=resolution,
        T_target_source=T_target_source,
    )


def parse_lidar_calibration(path, left_topic):
    path = Path(path).expanduser().resolve()
    calibration = load_camera_lidar_json(path)
    if left_topic and calibration.camera.topic != left_topic:
        raise argparse.ArgumentTypeError(
            f"{path}: target topic is {calibration.camera.topic!r}, "
            f"expected {left_topic!r}")
    return LidarProjectionCalibration(
        label="lidar",
        path=path,
        source="lidar",
        target_intrinsics=calibration.camera.intrinsics,
        target_distortion=calibration.camera.distortion,
        target_topic=calibration.camera.topic,
        T_target_source=calibration.T_camera_lidar,
    )


def bag_topics(bag):
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        return {connection.topic for connection in reader.connections}


def select_sources(topics, requested):
    available = {source for source, topic in DEFAULT_DEPTH_TOPICS.items()
                 if topic in topics}
    selected = (available if requested == "auto" else
                {"d455", "d435"} if requested == "both" else
                {requested})
    missing = selected - available
    if missing:
        raise SystemExit(
            f"Requested RealSense source(s) absent from bag: {sorted(missing)}; "
            f"available={sorted(available)}")
    if not selected:
        raise SystemExit("No D455 or D435 depth topic found")
    return tuple(source for source in ("d455", "d435") if source in selected)


def parse_requested_sources(value):
    if value == "auto":
        return None
    selected = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = selected - {"d455", "d435", "lidar"}
    if unknown or not selected:
        raise argparse.ArgumentTypeError(
            "--sources must be auto or a comma-separated subset of "
            "d455,d435,lidar")
    return selected


def read_target_frames(bag, topic, max_frames):
    frames = []
    print(f"[read camera] topic={topic}", flush=True)
    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise SystemExit(f"Target image topic not found: {topic}")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            image = image_to_numpy(message)
            if image.ndim == 2:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif str(message.encoding).lower() == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frames.append((stamp_to_sec(message), np.ascontiguousarray(image)))
            if len(frames) % 50 == 0:
                print(f"[read camera] {len(frames)} frames", flush=True)
    if not frames:
        raise SystemExit(f"No messages read from {topic}")
    if max_frames > 0 and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[index] for index in indices]
    print(f"[read camera] done; selected={len(frames)}", flush=True)
    return frames


def read_nearest_depths(bag, targets, sources, sync_tol):
    topic_to_source = {DEFAULT_DEPTH_TOPICS[source]: source for source in sources}
    result = {source: [None] * len(targets) for source in sources}
    deltas = {source: np.full(len(targets), np.nan) for source in sources}
    states = {topic: {"index": 0, "previous": None} for topic in topic_to_source}
    read_counts = {topic: 0 for topic in topic_to_source}
    print(f"[sync RealSense] sources={','.join(sources) or 'none'}", flush=True)

    def retain(topic, index, item):
        if item is None:
            return
        stamp, message = item
        delta = stamp - targets[index]
        if abs(delta) > sync_tol:
            return
        source = topic_to_source[topic]
        result[source][index] = image_to_numpy(message).astype(np.float32)
        deltas[source][index] = delta

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic in topic_to_source]
        for connection, _, raw in reader.messages(connections=connections):
            read_counts[connection.topic] += 1
            if read_counts[connection.topic] % 1000 == 0:
                print(
                    f"[sync RealSense] {connection.topic}: "
                    f"{read_counts[connection.topic]} messages",
                    flush=True)
            state = states[connection.topic]
            if state["index"] >= len(targets):
                continue
            message = reader.deserialize(raw, connection.msgtype)
            current = (stamp_to_sec(message), message)
            while (state["index"] < len(targets)
                   and targets[state["index"]] <= current[0]):
                previous = state["previous"]
                chosen = current if previous is None else min(
                    (previous, current),
                    key=lambda item: abs(item[0] - targets[state["index"]]))
                retain(connection.topic, state["index"], chosen)
                state["index"] += 1
            state["previous"] = current
            if all(state["index"] >= len(targets)
                   for state in states.values()):
                break
    for topic, state in states.items():
        while state["index"] < len(targets):
            retain(topic, state["index"], state["previous"])
            state["index"] += 1
    matched = {
        source: sum(item is not None for item in result[source])
        for source in sources
    }
    print(f"[sync RealSense] done; matched={matched}", flush=True)
    return result, deltas


def read_nearest_pointclouds(bag, targets, topic, sync_tol):
    result = [None] * len(targets)
    deltas = np.full(len(targets), np.nan)
    index = 0
    previous = None
    read_count = 0
    print(f"[sync LiDAR] topic={topic}", flush=True)

    def retain(target_index, item):
        if item is None:
            return
        stamp, message = item
        delta = stamp - targets[target_index]
        if abs(delta) <= sync_tol:
            result[target_index] = message
            deltas[target_index] = delta

    with AnyReader([bag], default_typestore=TYPESTORE) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return result, deltas
        for connection, _, raw in reader.messages(connections=connections):
            read_count += 1
            if read_count % 500 == 0:
                print(f"[sync LiDAR] {read_count} clouds", flush=True)
            if index >= len(targets):
                break
            message = reader.deserialize(raw, connection.msgtype)
            current = (stamp_to_sec(message), message)
            while index < len(targets) and targets[index] <= current[0]:
                chosen = current if previous is None else min(
                    (previous, current),
                    key=lambda item: abs(item[0] - targets[index]))
                retain(index, chosen)
                index += 1
            previous = current
    while index < len(targets):
        retain(index, previous)
        index += 1
    matched = sum(message is not None for message in result)
    print(
        f"[sync LiDAR] done; read={read_count}, matched={matched}/{len(targets)}",
        flush=True)
    return result, deltas


_POINTFIELD_DTYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2",
    5: "i4", 6: "u4", 7: "f4", 8: "f8",
}


def pointcloud2_xyz(message):
    """Decode XYZ from an organized or unorganized PointCloud2 message."""
    fields = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"} - fields.keys()
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {sorted(missing)}")
    endian = ">" if message.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for name in ("x", "y", "z"):
        field = fields[name]
        if field.datatype not in _POINTFIELD_DTYPES:
            raise ValueError(
                f"unsupported PointField datatype {field.datatype} for {name}")
        names.append(name)
        formats.append(endian + _POINTFIELD_DTYPES[field.datatype])
        offsets.append(int(field.offset))
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(message.point_step),
    })
    raw = memoryview(message.data)
    cloud = np.ndarray(
        shape=(int(message.height), int(message.width)), dtype=dtype,
        buffer=raw, strides=(int(message.row_step), int(message.point_step)))
    xyz = np.column_stack([
        np.asarray(cloud[name], dtype=np.float32).reshape(-1)
        for name in ("x", "y", "z")
    ])
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def project_lidar(message, calibration, output_shape):
    points = pointcloud2_xyz(message).astype(np.float64, copy=False)
    transform = calibration.T_target_source
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    positive = camera_points[:, 2] > 0
    camera_points = camera_points[positive]
    height, width = output_shape
    depth = np.full((height, width), np.nan, np.float32)
    if not len(camera_points):
        return depth
    fx, fy, cx, cy = calibration.target_intrinsics
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64)
    pixels, _ = cv2.projectPoints(
        camera_points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        camera_matrix, calibration.target_distortion)
    pixels = np.rint(pixels.reshape(-1, 2)).astype(np.int64)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    pixels = pixels[inside]
    z = camera_points[inside, 2].astype(np.float32)
    flat = np.full(height * width, np.inf, np.float32)
    indices = pixels[:, 1] * width + pixels[:, 0]
    np.minimum.at(flat, indices, z)
    flat[~np.isfinite(flat)] = np.nan
    return flat.reshape(height, width)


def depth_color(depth, dmin, dmax):
    valid = np.isfinite(depth) & (depth >= dmin) & (depth <= dmax)
    normalized = np.zeros(depth.shape, np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - dmin) * 255.0 / max(dmax - dmin, 1e-6),
        0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO), valid


def depth_edges(depth, valid):
    edge = np.zeros(depth.shape, dtype=bool)
    left_valid = valid[:, 1:] & valid[:, :-1]
    up_valid = valid[1:, :] & valid[:-1, :]
    dx = np.abs(depth[:, 1:] - depth[:, :-1])
    dy = np.abs(depth[1:, :] - depth[:-1, :])
    x_threshold = np.maximum(0.08, 0.05 * np.minimum(
        depth[:, 1:], depth[:, :-1]))
    y_threshold = np.maximum(0.08, 0.05 * np.minimum(
        depth[1:, :], depth[:-1, :]))
    edge[:, 1:] |= left_valid & (dx > x_threshold)
    edge[1:, :] |= up_valid & (dy > y_threshold)
    return edge


def rgb_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 60, 140) > 0


def edge_distance(depth_edge, image_edge):
    count = int(depth_edge.sum())
    if not count:
        return float("nan"), count
    distance = cv2.distanceTransform(
        (~image_edge).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[depth_edge].mean()), count


def label_panel(image, title, subtitle=""):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 40), (0, 0, 0), -1)
    cv2.putText(output, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(output, subtitle, (5, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (220, 220, 220), 1, cv2.LINE_AA)
    return output


def blend(base, color, valid, alpha):
    output = base.copy()
    output[valid] = cv2.addWeighted(
        base[valid], 1.0 - alpha, color[valid], alpha, 0)
    return output


def expand_sparse_visualization(color, valid, radius):
    """Make sparse projected points visible without changing metric masks."""
    radius = int(radius)
    if radius <= 0:
        return color, valid
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    expanded_valid = cv2.dilate(valid.astype(np.uint8), kernel) > 0
    expanded_color = cv2.dilate(color, kernel)
    return expanded_color, expanded_valid


def edge_panel(base, image_edge, projected_edge):
    output = (base.astype(np.float32) * 0.45).astype(np.uint8)
    output[image_edge] = (0, 255, 0)
    output[projected_edge] = (255, 0, 255)
    return output


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def pairwise_depth_statistics(first, second, depth_min, depth_max):
    valid = (np.isfinite(first) & np.isfinite(second)
             & (first >= depth_min) & (first <= depth_max)
             & (second >= depth_min) & (second <= depth_max))
    count = int(valid.sum())
    if not count:
        return count, float("nan"), float("nan"), float("nan")
    error = np.abs(first[valid] - second[valid])
    relative = error / np.maximum(np.minimum(first[valid], second[valid]), 1e-6)
    return (
        count,
        float(np.mean(error)),
        float(np.median(error)),
        float(np.median(relative)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare D455/D435/LiDAR projections directly from a bag")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument(
        "--calib", action="append", metavar="LABEL=PATH",
        help=("RealSense camchain; repeat for multiple sources/candidates. "
              "Default: CALIBRATION_PATHS['camera_d455']"))
    parser.add_argument(
        "--lidar-calib", type=Path,
        default=CALIBRATION_PATHS["camera_lidar"])
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--left-topic", default="/camera_2/image_rect")
    parser.add_argument(
        "--sources", default="auto",
        help="auto or comma-separated d455,d435,lidar")
    parser.add_argument(
        "--realsense", choices=("auto", "both", "d455", "d435"),
        default="auto", help="legacy RealSense-only source selector")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=20,
                        help="evenly sample this many target frames; 0 means all")
    parser.add_argument("--sync-tol", type=float, default=0.05)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    args.bag = args.bag.expanduser().resolve()
    if not args.bag.is_file():
        raise SystemExit(f"Bag not found: {args.bag}")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    topics = bag_topics(args.bag)
    requested_sources = parse_requested_sources(args.sources)
    if requested_sources is None and args.realsense != "auto":
        requested_sources = (
            {"d455", "d435"} if args.realsense == "both"
            else {args.realsense}
        )
    available_sources = {
        source for source, topic in DEFAULT_DEPTH_TOPICS.items()
        if topic in topics
    }
    if args.lidar_topic in topics:
        available_sources.add("lidar")
    if requested_sources is None:
        selected_sources = available_sources
    else:
        missing = requested_sources - available_sources
        if missing:
            raise SystemExit(
                f"Requested source(s) absent from bag: {sorted(missing)}; "
                f"available={sorted(available_sources)}")
        selected_sources = requested_sources

    real_sources = tuple(
        source for source in ("d455", "d435") if source in selected_sources)
    calibration_specs = list(args.calib or [])
    if not calibration_specs and "d455" in real_sources:
        default_path = CALIBRATION_PATHS["camera_d455"]
        if default_path is not None:
            calibration_specs.append(f"d455={default_path}")
    calibrations = [parse_calibration(spec, args.left_topic)
                    for spec in calibration_specs]
    calibrations = [calib for calib in calibrations
                    if calib.source in real_sources]
    calibrated_real_sources = {calib.source for calib in calibrations}
    missing_calibrations = set(real_sources) - calibrated_real_sources
    if missing_calibrations:
        message = ("No calibration supplied for RealSense source(s): "
                   f"{sorted(missing_calibrations)}")
        if requested_sources is not None:
            raise SystemExit(message)
        print(f"Warning: {message}; skipping them")
        real_sources = tuple(
            source for source in real_sources
            if source in calibrated_real_sources)

    lidar_calibration = None
    if "lidar" in selected_sources:
        if args.lidar_calib and args.lidar_calib.expanduser().is_file():
            lidar_calibration = parse_lidar_calibration(
                args.lidar_calib, args.left_topic)
        elif requested_sources is not None:
            raise SystemExit(
                f"LiDAR calibration not found: {args.lidar_calib}")
        else:
            print(f"Warning: LiDAR calibration not found: {args.lidar_calib}; "
                  "skipping LiDAR")
            selected_sources.discard("lidar")
    if not calibrations and lidar_calibration is None:
        raise SystemExit("No available source has a usable calibration")
    duplicate_labels = {calib.label for calib in calibrations
                        if sum(other.label == calib.label
                               for other in calibrations) > 1}
    if duplicate_labels:
        raise SystemExit(f"Calibration labels must be unique: {sorted(duplicate_labels)}")

    print("[stage 1/4] Reading target camera frames...", flush=True)
    frames = read_target_frames(args.bag, args.left_topic, args.max_files)
    targets = np.asarray([stamp for stamp, _ in frames], np.float64)
    print("[stage 2/4] Matching RealSense depth...", flush=True)
    depths, deltas = read_nearest_depths(
        args.bag, targets, real_sources, args.sync_tol)
    if lidar_calibration is not None:
        print("[stage 3/4] Matching LiDAR clouds...", flush=True)
        lidar_messages, lidar_deltas = read_nearest_pointclouds(
            args.bag, targets, args.lidar_topic, args.sync_tol)
    else:
        lidar_messages = [None] * len(targets)
        lidar_deltas = np.full(len(targets), np.nan)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0

    print(f"Bag: {args.bag}", flush=True)
    print(f"Target: {args.left_topic}; sampled frames={len(frames)}")
    active_sources = list(real_sources)
    if lidar_calibration is not None:
        active_sources.append("lidar")
    print(f"Depth source(s): {', '.join(active_sources)}")
    for calib in calibrations:
        print(f"  {calib.label}: {calib.source}, {calib.path}")
    if lidar_calibration is not None:
        print(f"  lidar: {args.lidar_topic}, {lidar_calibration.path}")

    print("[stage 4/4] Projecting sensors and writing montages...", flush=True)
    for frame_index, (target_stamp, image) in enumerate(frames):
        print(
            f"[frame {frame_index + 1}/{len(frames)}] "
            f"stamp={target_stamp:.9f}",
            flush=True)
        image_h, image_w = image.shape[:2]
        image_edge = rgb_edges(image)
        top = [label_panel(image, "Target RGB", f"stamp={target_stamp:.9f}")]
        rgb_edge_view = np.zeros_like(image)
        rgb_edge_view[image_edge] = (0, 255, 0)
        bottom = [label_panel(rgb_edge_view, "RGB edges", "green=RGB edge")]
        usable = False
        projected_by_source = {}

        for calib in calibrations:
            raw_depth = depths[calib.source][frame_index]
            if raw_depth is None:
                top.append(label_panel(np.zeros_like(image), calib.label,
                                       "no synchronized depth"))
                bottom.append(np.zeros_like(image))
                continue
            expected_w, expected_h = calib.target_resolution
            if (expected_h, expected_w) != (image_h, image_w):
                raise SystemExit(
                    f"{calib.label}: YAML target is {expected_w}x{expected_h}, "
                    f"bag image is {image_w}x{image_h}")
            projected = step2_ir_depth_to_L(
                raw_depth * args.depth_scale,
                calib.source_intrinsics,
                calib.T_target_source,
                calib.target_intrinsics,
                calib.target_distortion,
                (image_h, image_w),
                splat=True,
            )
            projected_by_source[calib.source] = projected
            color, valid = depth_color(
                projected, args.depth_min, args.depth_max)
            projected_edge = depth_edges(projected, valid)
            distance, edge_pixels = edge_distance(projected_edge, image_edge)
            coverage = float(valid.mean())
            delta = float(deltas[calib.source][frame_index])
            overlay = blend(image, color, valid, args.alpha)
            top.append(label_panel(
                overlay, f"{calib.label} ({calib.source.upper()})",
                f"cover={coverage:.1%}, dt={delta * 1e3:+.1f} ms"))
            bottom.append(label_panel(
                edge_panel(image, image_edge, projected_edge),
                f"{calib.label} depth edges",
                f"edge distance={distance:.2f}px, n={edge_pixels}"))
            rows.append({
                "row_type": "frame",
                "frame": frame_index,
                "stamp": f"{target_stamp:.9f}",
                "label": calib.label,
                "source": calib.source,
                "sync_delta_ms": delta * 1e3,
                "coverage": coverage,
                "depth_edge_pixels": edge_pixels,
                "edge_distance_px": distance,
                "median_edge_distance_px": "",
                "pairwise_wins": "",
                "compared_frames": "",
            })
            usable = True

        if lidar_calibration is not None:
            message = lidar_messages[frame_index]
            if message is None:
                top.append(label_panel(
                    np.zeros_like(image), "LiDAR", "no synchronized cloud"))
                bottom.append(np.zeros_like(image))
            else:
                projected = project_lidar(
                    message, lidar_calibration, (image_h, image_w))
                projected_by_source["lidar"] = projected
                color, valid = depth_color(
                    projected, args.depth_min, args.depth_max)
                projected_edge = depth_edges(projected, valid)
                distance, edge_pixels = edge_distance(
                    projected_edge, image_edge)
                coverage = float(valid.mean())
                delta = float(lidar_deltas[frame_index])
                vis_color, vis_valid = expand_sparse_visualization(
                    color, valid, LIDAR_VIS_RADIUS)
                overlay = blend(image, vis_color, vis_valid, args.alpha)
                top.append(label_panel(
                    overlay, "LiDAR",
                    f"cover={coverage:.1%}, dt={delta * 1e3:+.1f} ms, "
                    f"vis r={LIDAR_VIS_RADIUS}px"))
                bottom.append(label_panel(
                    edge_panel(image, image_edge, projected_edge),
                    "LiDAR depth edges",
                    f"edge distance={distance:.2f}px, n={edge_pixels}"))
                rows.append({
                    "row_type": "frame",
                    "frame": frame_index,
                    "stamp": f"{target_stamp:.9f}",
                    "label": "lidar",
                    "source": "lidar",
                    "sync_delta_ms": delta * 1e3,
                    "coverage": coverage,
                    "depth_edge_pixels": edge_pixels,
                    "edge_distance_px": distance,
                    "median_edge_distance_px": "",
                    "pairwise_wins": "",
                    "compared_frames": "",
                })
                usable = True

        source_names = [
            source for source in ("d455", "d435", "lidar")
            if source in projected_by_source
        ]
        for first_index, first in enumerate(source_names):
            for second in source_names[first_index + 1:]:
                count, mae, median_ae, median_relative = (
                    pairwise_depth_statistics(
                        projected_by_source[first],
                        projected_by_source[second],
                        args.depth_min, args.depth_max))
                rows.append({
                    "row_type": "pairwise",
                    "frame": frame_index,
                    "stamp": f"{target_stamp:.9f}",
                    "label": f"{first}_vs_{second}",
                    "source": f"{first}_vs_{second}",
                    "overlap_pixels": count,
                    "depth_mae_m": mae,
                    "depth_median_ae_m": median_ae,
                    "median_relative_error": median_relative,
                })

        if not usable:
            print(f"Skip frame {frame_index}: no synchronized selected depth")
            continue
        montage = np.vstack([np.hstack(top), np.hstack(bottom)])
        output = args.out_dir / f"sample_{frame_index:04d}.jpg"
        cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
        written += 1
        print(f"[frame {frame_index + 1}/{len(frames)}] wrote {output}",
              flush=True)

    summary_rows = []
    finite_by_frame = {}
    for row in rows:
        if row["row_type"] != "frame":
            continue
        distance = float(row["edge_distance_px"])
        if np.isfinite(distance):
            finite_by_frame.setdefault(int(row["frame"]), {})[
                row["label"]] = distance
    summary_sources = list(calibrations)
    if lidar_calibration is not None:
        summary_sources.append(lidar_calibration)
    for calib in summary_sources:
        selected = [row for row in rows
                    if row["row_type"] == "frame"
                    and row["label"] == calib.label]
        if not selected:
            continue
        distances = np.asarray(
            [row["edge_distance_px"] for row in selected], np.float64)
        finite_distances = distances[np.isfinite(distances)]
        wins = sum(
            calib.label in values
            and values[calib.label] == min(values.values())
            for values in finite_by_frame.values()
        )
        summary = {
            "row_type": "average",
            "frame": "",
            "stamp": "",
            "label": calib.label,
            "source": calib.source,
            "sync_delta_ms": float(np.mean([
                abs(row["sync_delta_ms"]) for row in selected])),
            "coverage": float(np.mean([row["coverage"] for row in selected])),
            "depth_edge_pixels": int(round(np.mean([
                row["depth_edge_pixels"] for row in selected]))),
            "edge_distance_px": float(np.mean(finite_distances)),
            "median_edge_distance_px": float(np.median(finite_distances)),
            "pairwise_wins": wins,
            "compared_frames": len(finite_by_frame),
        }
        summary_rows.append(summary)
        print(f"{calib.label}: coverage={summary['coverage']:.1%}, "
              f"edge_distance mean={summary['edge_distance_px']:.3f}px, "
              f"median={summary['median_edge_distance_px']:.3f}px, "
              f"wins={wins}/{len(finite_by_frame)}, "
              f"mean_abs_dt={summary['sync_delta_ms']:.2f}ms")
    pairwise_labels = sorted({
        row["label"] for row in rows if row["row_type"] == "pairwise"
    })
    for label in pairwise_labels:
        selected = [row for row in rows
                    if row["row_type"] == "pairwise"
                    and row["label"] == label]
        valid = [row for row in selected
                 if int(row["overlap_pixels"]) > 0
                 and np.isfinite(float(row["depth_mae_m"]))]
        if not valid:
            continue
        summary = {
            "row_type": "pairwise_average",
            "frame": "",
            "stamp": "",
            "label": label,
            "source": label,
            "overlap_pixels": int(sum(
                int(row["overlap_pixels"]) for row in valid)),
            "depth_mae_m": float(np.mean([
                float(row["depth_mae_m"]) for row in valid])),
            "depth_median_ae_m": float(np.median([
                float(row["depth_median_ae_m"]) for row in valid])),
            "median_relative_error": float(np.median([
                float(row["median_relative_error"]) for row in valid])),
            "compared_frames": len(valid),
        }
        summary_rows.append(summary)
        print(f"{label}: overlap={summary['overlap_pixels']} px, "
              f"frame-mean MAE={summary['depth_mae_m']:.3f}m, "
              f"median relative={summary['median_relative_error']:.1%}, "
              f"frames={len(valid)}")
    write_csv(args.out_dir / "comparison_metrics.csv", rows + summary_rows)
    print(f"Done: {written}/{len(frames)} montages -> {args.out_dir}")


if __name__ == "__main__":
    main()
