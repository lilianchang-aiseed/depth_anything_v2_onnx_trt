#!/usr/bin/env python3
"""Read unmodified Kalibr outputs and normalize transform directions.

The public helpers always return transforms in the directions used by the GT
pipeline, regardless of whether Kalibr placed a camera in cam0 or cam1:

    load_left_d455(...) -> T_left_d455   (T_Left<-D455)
    load_d455_d435(...) -> T_d455_d435   (T_D455<-D435)

Both Kalibr ``*-camchain.yaml`` and ``*-results-cam.txt`` are accepted.
"""

from dataclasses import dataclass
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
