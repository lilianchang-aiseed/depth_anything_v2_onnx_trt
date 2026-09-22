#!/usr/bin/env python3
"""Read camera/RealSense/LiDAR calibrations and normalize matrix directions.

The public helpers always return transforms in the directions used by the GT
pipeline, regardless of whether Kalibr placed a camera in cam0 or cam1:

    load_left_d455(...) -> T_left_d455   (T_Left<-D455)
    load_d455_d435(...) -> T_d455_d435   (T_D455<-D435)

Both Kalibr ``*-camchain.yaml`` and ``*-results-cam.txt`` are accepted.

The LiDAR JSON helper follows the same ``T_target_source`` convention.  A
stored ``T_lidar_camera`` is therefore inverted and returned as
``T_camera_lidar``.
"""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re

import numpy as np
import yaml


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Camera:
    topic: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    camera_model: str = "pinhole"
    distortion_model: str = "radtan"


@dataclass(frozen=True)
class Chain:
    cam0: Camera
    cam1: Camera
    T_cam1_cam0: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class LeftD455Calibration:
    left: Camera
    d455: Camera
    T_left_d455: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class D455D435Calibration:
    d455: Camera
    d435: Camera
    T_d455_d435: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class CameraLidarCalibration:
    camera: Camera
    T_lidar_camera: np.ndarray
    T_camera_lidar: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class ProjectionMatrices:
    """All available transforms, expressed as ``T_target_source``."""

    T_camera_d455: np.ndarray
    T_d455_d435: np.ndarray | None
    T_camera_d435: np.ndarray | None
    T_camera_lidar: np.ndarray | None


@dataclass(frozen=True)
class MultiCameraLidarTransforms:
    """Raw/rectified camera coordinate maps relative to a LiDAR frame.

    ``H`` is used instead of ``T`` when a map may contain the image reflection
    introduced by the deployed rectifier's row-axis convention.
    """

    reference_camera: int
    camera_lidar_path: Path
    camera_rig_path: Path
    cameras: dict


def _numbers(text):
    return [float(value) for value in _NUMBER.findall(text)]


def _quat_xyzw_to_rotation(q):
    """Hamilton xyzw quaternion to rotation; Kalibr TXT stores JPL, see caller."""
    x, y, z, w = np.asarray(q, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError("zero-length quaternion in Kalibr result")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _make_transform(rotation, translation):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _validate_transform(name, transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} has invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation determinant is not +1")
    return transform


def _validate_coordinate_map(name, transform):
    """Validate an orthogonal homogeneous map, allowing determinant -1."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} has invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation/reflection block is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(abs(determinant), 1.0, atol=2e-3):
        raise ValueError(f"{name} determinant is not +/-1: {determinant}")
    return transform


def _camera_from_mapping(mapping):
    return Camera(
        topic=str(mapping.get("rostopic", "")),
        intrinsics=np.asarray(mapping.get("intrinsics", []), dtype=float),
        distortion=np.asarray(mapping.get("distortion_coeffs", []), dtype=float),
        camera_model=str(mapping.get("camera_model", "pinhole")),
        distortion_model=str(mapping.get("distortion_model", "radtan")),
    )


def _load_yaml(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = "\n".join(text.splitlines()[1:])
    data = yaml.safe_load(text) or {}
    if "cam0" not in data or "cam1" not in data:
        raise ValueError(f"{path}: expected cam0 and cam1")
    if "T_cn_cnm1" not in data["cam1"]:
        raise ValueError(f"{path}: cam1.T_cn_cnm1 is missing")
    return Chain(
        cam0=_camera_from_mapping(data["cam0"]),
        cam1=_camera_from_mapping(data["cam1"]),
        T_cam1_cam0=_validate_transform(
            "T_cam1_cam0", data["cam1"]["T_cn_cnm1"]),
        source_path=path,
    )


def _load_txt(path):
    cameras = {"cam0": {}, "cam1": {}}
    current = None
    quaternion = translation = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(cam[01])\s*\(([^)]+)\):", line)
        if match:
            current = match.group(1)
            cameras[current]["rostopic"] = match.group(2)
            continue
        if current and line.startswith("distortion:"):
            cameras[current]["distortion_coeffs"] = _numbers(line)[:4]
        elif current and line.startswith("projection:"):
            cameras[current]["intrinsics"] = _numbers(line)[:4]
        elif line.startswith("q:"):
            quaternion = _numbers(line)[:4]
        elif line.startswith("t:"):
            translation = _numbers(line)[:3]
    if quaternion is None or translation is None:
        raise ValueError(f"{path}: baseline q/t is missing")
    for name in ("cam0", "cam1"):
        if "rostopic" not in cameras[name] or "intrinsics" not in cameras[name]:
            raise ValueError(f"{path}: incomplete {name} calibration")
    # Kalibr/aslam serializes the TXT quaternion in JPL convention. The
    # transpose makes it agree with the matrix in the corresponding YAML.
    transform = _make_transform(
        _quat_xyzw_to_rotation(quaternion).T, translation)
    return Chain(
        cam0=_camera_from_mapping(cameras["cam0"]),
        cam1=_camera_from_mapping(cameras["cam1"]),
        T_cam1_cam0=_validate_transform("T_cam1_cam0", transform),
        source_path=path,
    )


def load_kalibr_chain(path):
    """Load a raw Kalibr camchain YAML or human-readable results TXT."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Kalibr calibration not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    if suffix == ".txt":
        return _load_txt(path)
    raise ValueError(f"unsupported Kalibr file type: {path}")


def _role(topic):
    topic = topic.lower()
    if "stereo" in topic and "/left" in topic:
        return "left"
    if re.fullmatch(r"/camera_[0-3]/image_rect", topic):
        return "left"
    if "d455" in topic and ("infra1" in topic or "/depth/" in topic):
        return "d455"
    if "d435" in topic and ("infra1" in topic or "/depth/" in topic):
        return "d435"
    if "d455" in topic and "color" in topic:
        return "d455"
    if "d435" in topic and "color" in topic:
        return "d435"
    raise ValueError(f"cannot identify camera role from topic: {topic}")


def _camera_for_role(chain, role):
    matches = [camera for camera in (chain.cam0, chain.cam1)
               if _role(camera.topic) == role]
    if len(matches) != 1:
        raise ValueError(
            f"{chain.source_path}: expected one {role} camera, got {len(matches)}")
    return matches[0]


def _transform_target_source(chain, target_role, source_role):
    role0 = _role(chain.cam0.topic)
    role1 = _role(chain.cam1.topic)
    if role0 == source_role and role1 == target_role:
        result = chain.T_cam1_cam0
    elif role0 == target_role and role1 == source_role:
        result = np.linalg.inv(chain.T_cam1_cam0)
    else:
        raise ValueError(
            f"{chain.source_path}: expected {source_role}<->{target_role}, "
            f"got {role0}<->{role1}")
    return _validate_transform(f"T_{target_role}_{source_role}", result)


def load_left_d455(path):
    """Return a left/D455 calibration normalized to T_Left<-D455."""
    chain = load_kalibr_chain(path)
    return LeftD455Calibration(
        left=_camera_for_role(chain, "left"),
        d455=_camera_for_role(chain, "d455"),
        T_left_d455=_transform_target_source(chain, "left", "d455"),
        source_path=chain.source_path,
    )


def load_d455_d435(path):
    """Return a D455/D435 calibration normalized to T_D455<-D435."""
    chain = load_kalibr_chain(path)
    return D455D435Calibration(
        d455=_camera_for_role(chain, "d455"),
        d435=_camera_for_role(chain, "d435"),
        T_d455_d435=_transform_target_source(chain, "d455", "d435"),
        source_path=chain.source_path,
    )


def load_camera_lidar_json(path):
    """Load cam/LiDAR JSON and return the transform as ``T_camera<-lidar``.

    The JSON vector is ``[tx, ty, tz, qx, qy, qz, qw]`` and its stored
    ``T_lidar_camera`` maps camera-frame points into the LiDAR frame.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"camera/LiDAR calibration not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_data = data.get("camera", {})
    vector = np.asarray(
        data.get("results", {}).get("T_lidar_camera", []), dtype=float)
    if vector.shape != (7,):
        raise ValueError(
            f"{path}: results.T_lidar_camera must contain tx ty tz qx qy qz qw")
    T_lidar_camera = _validate_transform(
        "T_lidar_camera",
        _make_transform(_quat_xyzw_to_rotation(vector[3:]), vector[:3]),
    )
    T_camera_lidar = _validate_transform(
        "T_camera_lidar", np.linalg.inv(T_lidar_camera))
    camera = Camera(
        topic=str(data.get("meta", {}).get("image_topic", "")),
        intrinsics=np.asarray(camera_data.get("intrinsics", []), dtype=float),
        distortion=np.asarray(
            camera_data.get("distortion_coeffs", []), dtype=float),
        camera_model=str(camera_data.get("camera_model", "pinhole")),
        distortion_model="radtan",
    )
    if camera.intrinsics.shape != (4,):
        raise ValueError(f"{path}: camera.intrinsics must contain fx fy cx cy")
    return CameraLidarCalibration(
        camera=camera,
        T_lidar_camera=T_lidar_camera,
        T_camera_lidar=T_camera_lidar,
        source_path=path,
    )


def load_projection_matrices(left_d455_path, d455_d435_path=None,
                             camera_lidar_path=None):
    """Load and compose every available transform required for projection."""
    left_d455 = load_left_d455(left_d455_path)
    d455_d435 = (load_d455_d435(d455_d435_path)
                 if d455_d435_path else None)
    camera_lidar = (load_camera_lidar_json(camera_lidar_path)
                    if camera_lidar_path else None)
    T_camera_d435 = None
    if d455_d435 is not None:
        T_camera_d435 = _validate_transform(
            "T_camera_d435",
            left_d455.T_left_d455 @ d455_d435.T_d455_d435,
        )
    return ProjectionMatrices(
        T_camera_d455=left_d455.T_left_d455,
        T_d455_d435=(None if d455_d435 is None
                     else d455_d435.T_d455_d435),
        T_camera_d435=T_camera_d435,
        T_camera_lidar=(None if camera_lidar is None
                        else camera_lidar.T_camera_lidar),
    )


def _left_from_right_pair_transform(pair):
    """Match the physical pair transform used by the deployed v3.1 rectifier."""
    rotation_yaml = np.asarray(pair["cam1"]["R"], dtype=float)
    translation_yaml = np.asarray(pair["cam1"]["t"], dtype=float)
    transform = _make_transform(
        rotation_yaml.T,
        rotation_yaml @ translation_yaml,
    )
    return _validate_transform("T_left_raw_right_raw", transform)


def _epipolar_plane(rotation, translation):
    """Match ``rectify_utils.epiploar_planes_from_extrinsics`` for cam0."""
    z0 = np.array([0.0, 0.0, 1.0])
    z1_in_0 = np.asarray(rotation, dtype=float) @ z0
    dot_z0_t = float(np.dot(z0, translation))
    dot_z1_t = float(np.dot(z1_in_0, translation))
    if abs(dot_z0_t) > 1e-10:
        normal = (-dot_z1_t / dot_z0_t) * z0 + z1_in_0
    else:
        normal = z0.copy()
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("degenerate epipolar-plane normal")
    return normal / norm


def _rect_from_raw_coordinate_map(pair):
    """Return the exact v3.1 left-image coordinate map ``H_rect<-raw``.

    v3.1 generates output rows from ``+v`` to ``-v`` and therefore defines
    image-down as ``-v``.  This can make the raw/rect coordinate map a
    reflection (determinant -1); preserving it is required to match pixels.
    """
    T_left_right = _left_from_right_pair_transform(pair)
    rotation = T_left_right[:3, :3]
    translation = T_left_right[:3, 3]
    normal = _epipolar_plane(rotation, translation)
    right = -translation
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-12:
        raise ValueError("stereo baseline is too small to define rectification")
    right /= right_norm
    image_v = np.cross(normal, right)
    image_v /= np.linalg.norm(image_v)
    if float(np.dot(right, [1.0, 0.0, 0.0])) < 0.0:
        right = -right
    down = -image_v
    H_raw_rect = _make_transform(
        np.column_stack((right, down, normal)), np.zeros(3))
    H_raw_rect = _validate_coordinate_map("H_raw_rect", H_raw_rect)
    return _validate_coordinate_map("H_rect_raw", np.linalg.inv(H_raw_rect))


def _pair_graph(rig_data):
    graph = {}
    pair_for_left = {}
    pattern = re.compile(r"^cam_pair_(\d+)_(\d+)$")
    for key, pair in rig_data.items():
        match = pattern.fullmatch(str(key))
        if match is None:
            continue
        left, right = map(int, match.groups())
        if left in pair_for_left:
            raise ValueError(f"multiple rectification pairs use camera {left} as left")
        pair_for_left[left] = pair
        T_left_right = _left_from_right_pair_transform(pair)
        graph.setdefault(left, {})[right] = T_left_right
        graph.setdefault(right, {})[left] = np.linalg.inv(T_left_right)
    if not graph:
        raise ValueError("camera rig YAML contains no cam_pair_<left>_<right>")
    return graph, pair_for_left


def _transforms_to_reference(graph, reference):
    """Return ``T_reference<-camera`` for every camera in a connected graph."""
    if reference not in graph:
        raise ValueError(f"reference camera {reference} is absent from camera rig")
    result = {reference: np.eye(4)}
    queue = [reference]
    while queue:
        target = queue.pop(0)
        T_reference_target = result[target]
        for source, T_target_source in graph[target].items():
            candidate = T_reference_target @ T_target_source
            if source not in result:
                result[source] = _validate_transform(
                    f"T_camera{reference}_camera{source}", candidate)
                queue.append(source)
            # A four-camera ring is over-constrained and independently
            # calibrated pair transforms do not close perfectly. Keep the
            # first (shortest BFS) path instead of silently averaging SE(3).
    if len(result) != len(graph):
        missing = sorted(set(graph) - result.keys())
        raise ValueError(f"camera rig is disconnected; missing cameras {missing}")
    return result


def load_multicamera_lidar_transforms(camera_lidar_path, camera_rig_path):
    """Compose all ``camera_[0-3]`` raw/rect coordinate maps with LiDAR."""
    camera_lidar = load_camera_lidar_json(camera_lidar_path)
    match = re.fullmatch(
        r"/camera_(\d+)/image_rect", camera_lidar.camera.topic)
    if match is None:
        raise ValueError(
            "LiDAR JSON image_topic must be /camera_<num>/image_rect; got "
            f"{camera_lidar.camera.topic!r}")
    reference = int(match.group(1))
    camera_rig_path = Path(camera_rig_path).expanduser().resolve()
    if not camera_rig_path.is_file():
        raise FileNotFoundError(f"camera rig calibration not found: {camera_rig_path}")
    rig_data = yaml.safe_load(camera_rig_path.read_text(encoding="utf-8")) or {}
    graph, pair_for_left = _pair_graph(rig_data)
    T_reference_raw_from_raw = _transforms_to_reference(graph, reference)
    missing_rect = sorted(set(graph) - pair_for_left.keys())
    if missing_rect:
        raise ValueError(
            f"no left-view rectification pair for cameras {missing_rect}")

    H_rect_raw = {
        index: _rect_from_raw_coordinate_map(pair_for_left[index])
        for index in sorted(graph)
    }
    # JSON stores T_lidar<-reference_rect.  First bridge its exact deployed
    # rect coordinates to reference raw, then traverse physical pair extrinsics.
    T_lidar_reference_rect = camera_lidar.T_lidar_camera
    H_lidar_reference_raw = _validate_coordinate_map(
        "H_lidar_reference_raw",
        T_lidar_reference_rect @ H_rect_raw[reference])

    cameras = {}
    for index in sorted(graph):
        T_reference_raw_camera_raw = T_reference_raw_from_raw[index]
        H_lidar_camera_raw = _validate_coordinate_map(
            f"H_lidar_camera_{index}_raw",
            H_lidar_reference_raw @ T_reference_raw_camera_raw)
        H_camera_raw_camera_rect = _validate_coordinate_map(
            f"H_camera_{index}_raw_camera_{index}_rect",
            np.linalg.inv(H_rect_raw[index]))
        H_lidar_camera_rect = _validate_coordinate_map(
            f"H_lidar_camera_{index}_rect",
            H_lidar_camera_raw @ H_camera_raw_camera_rect)
        cameras[index] = {
            "raw_topic": f"/camera_{index}/image_raw",
            "rect_topic": f"/camera_{index}/image_rect",
            "H_rect_from_raw": H_rect_raw[index],
            "H_lidar_from_raw": H_lidar_camera_raw,
            "H_raw_from_lidar": np.linalg.inv(H_lidar_camera_raw),
            "T_lidar_from_rect": H_lidar_camera_rect,
            "T_rect_from_lidar": np.linalg.inv(H_lidar_camera_rect),
            "raw_map_determinant": float(
                np.linalg.det(H_lidar_camera_raw[:3, :3])),
            "rect_map_determinant": float(
                np.linalg.det(H_lidar_camera_rect[:3, :3])),
        }
    return MultiCameraLidarTransforms(
        reference_camera=reference,
        camera_lidar_path=camera_lidar.source_path,
        camera_rig_path=camera_rig_path,
        cameras=cameras,
    )


def _serializable_multicamera(result):
    cameras = {}
    for index, values in result.cameras.items():
        cameras[f"camera_{index}"] = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in values.items()
        }
    return {
        "convention": "H_target_from_source",
        "reference_camera": result.reference_camera,
        "camera_lidar_calibration": str(result.camera_lidar_path),
        "camera_rig_calibration": str(result.camera_rig_path),
        "note": (
            "raw H matrices may have determinant -1 because they preserve the "
            "deployed rectifier's image-row reflection; rect T matrices are "
            "proper rigid transforms when rect_map_determinant is +1"
        ),
        "cameras": cameras,
    }


def write_multicamera_lidar_yaml(path, result):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.write_text(
        yaml.safe_dump(_serializable_multicamera(result), sort_keys=False),
        encoding="utf-8")
    return path


def _print_matrix(name, matrix):
    if matrix is None:
        print(f"{name}: unavailable")
        return
    print(f"{name}:")
    print(np.array2string(matrix, precision=9, suppress_small=True))


def main():
    parser = argparse.ArgumentParser(
        description="Normalize and print camera/RealSense/LiDAR transforms")
    parser.add_argument("--camera-d455", type=Path)
    parser.add_argument("--d455-d435", type=Path)
    parser.add_argument("--camera-lidar", type=Path)
    parser.add_argument(
        "--camera-rig", type=Path,
        help="four-pair Double-Sphere YAML used by the deployed rectifier")
    parser.add_argument(
        "--output-yaml", type=Path,
        help="write all raw/rect camera-to-LiDAR maps to this new YAML")
    args = parser.parse_args()
    if args.camera_d455:
        matrices = load_projection_matrices(
            args.camera_d455, args.d455_d435, args.camera_lidar)
        for name in (
            "T_camera_d455", "T_d455_d435", "T_camera_d435",
            "T_camera_lidar",
        ):
            _print_matrix(name, getattr(matrices, name))
    if args.camera_rig:
        if not args.camera_lidar:
            parser.error("--camera-rig requires --camera-lidar")
        result = load_multicamera_lidar_transforms(
            args.camera_lidar, args.camera_rig)
        print(f"reference camera: camera_{result.reference_camera}/image_rect")
        for index, values in result.cameras.items():
            print(f"camera_{index}:")
            _print_matrix("  H_lidar_from_raw", values["H_lidar_from_raw"])
            print(f"  raw determinant: {values['raw_map_determinant']:+.6f}")
            _print_matrix("  T_lidar_from_rect", values["T_lidar_from_rect"])
            print(f"  rect determinant: {values['rect_map_determinant']:+.6f}")
        if args.output_yaml:
            output = write_multicamera_lidar_yaml(args.output_yaml, result)
            print(f"wrote: {output}")
    elif args.output_yaml:
        parser.error("--output-yaml requires --camera-rig")
    if not args.camera_d455 and not args.camera_rig:
        parser.error("provide --camera-d455 and/or --camera-rig")


if __name__ == "__main__":
    main()
