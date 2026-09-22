#!/usr/bin/env python3
"""Reusable CPU Double-Sphere inverse-map rectification.

The geometry intentionally matches depth_anything_v2_trt_node-v3.1.py:
each physical camera uses the LEFT map of the configured stereo pair, output
rows are image-top to image-bottom, and the horizontal axis is kept consistent
with the source camera's image-right direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _epipolar_plane(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Match rectify_utils.epiploar_planes_from_extrinsics() for cam0."""
    z0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z1_in_0 = R @ z0
    dot_z0_t = float(np.dot(z0, t))
    dot_z1_t = float(np.dot(z1_in_0, t))
    if abs(dot_z0_t) > 1e-10:
        normal = (-dot_z1_t / dot_z0_t) * z0 + z1_in_0
    else:
        normal = z0.copy()
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Degenerate epipolar-plane normal")
    return normal / norm


def _plane_basis(normal: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the basis and orientation corrections used by the v3.1 node."""
    u_axis = -np.asarray(t, dtype=np.float64)
    u_norm = float(np.linalg.norm(u_axis))
    if u_norm < 1e-12:
        raise ValueError("Stereo baseline is too small to define rectification")
    u_axis /= u_norm
    v_axis = np.cross(normal, u_axis)
    v_norm = float(np.linalg.norm(v_axis))
    if v_norm < 1e-12:
        raise ValueError("Degenerate rectification plane basis")
    v_axis /= v_norm

    # Preserve source image-left/right, as in v3.1.
    if float(np.dot(u_axis, np.array([1.0, 0.0, 0.0]))) < 0.0:
        u_axis = -u_axis
    return u_axis, v_axis


def _inverse_map(
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    intrinsics: tuple[float, float, float, float, float, float],
    height: int,
    width: int,
    half_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build output-pixel -> source-pixel Double-Sphere maps."""
    xi, alpha, fx, fy, cx, cy = intrinsics
    if width >= height:
        half_width = float(np.tan(half_angle_rad))
        half_height = half_width * height / width
    else:
        half_height = float(np.tan(half_angle_rad))
        half_width = half_height * width / height

    grid_x = np.linspace(-half_width, half_width, width, dtype=np.float64)
    # Same orientation as v3.1: image top uses +v_axis.
    grid_y = np.linspace(half_height, -half_height, height, dtype=np.float64)
    uu, vv = np.meshgrid(grid_x, grid_y)
    rays = (
        normal[None, None, :]
        + uu[..., None] * u_axis[None, None, :]
        + vv[..., None] * v_axis[None, None, :]
    )

    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    k = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + k * k)
    denominator = alpha * d2 + (1.0 - alpha) * k

    w1 = alpha / (1.0 - alpha) if alpha <= 0.5 else (1.0 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    # Interpret FOV as a circular viewing cone around the rectified optical
    # axis. Pixels in the square corners lie outside that cone and remain
    # black, producing the requested circle-frame rectification.
    cone_radius = float(np.tan(half_angle_rad))
    cone_valid = (uu * uu + vv * vv) <= cone_radius * cone_radius
    valid = (z > -w2 * d1) & (denominator > 1e-9) & cone_valid

    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)
    map_x[valid] = (fx * x[valid] / denominator[valid] + cx).astype(np.float32)
    map_y[valid] = (fy * y[valid] / denominator[valid] + cy).astype(np.float32)
    return map_x, map_y, valid


@dataclass(frozen=True)
class CameraSpec:
    camera_id: int
    input_topic: str
    output_topic: str
    calibration_pair: str


class DoubleSphereRectifier:
    """Precomputed CPU rectifier for one physical camera/left stereo view."""

    def __init__(
        self,
        pair_config: dict[str, Any],
        output_height: int,
        output_width: int,
        half_angle_deg: float,
    ) -> None:
        cam0 = pair_config["cam0"]
        cam1 = pair_config["cam1"]
        calibration_width, calibration_height = map(int, cam0["resolution"])
        stream_width, stream_height = map(
            int,
            pair_config.get("stream_resolution", (calibration_width, calibration_height)),
        )
        scale_x = stream_width / calibration_width
        scale_y = stream_height / calibration_height
        if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Non-uniform calibration-to-stream scaling is unsupported: "
                f"scale_x={scale_x}, scale_y={scale_y}"
            )

        intr = cam0["intrinsics"]
        ds_intrinsics = (
            float(intr[0]),
            float(intr[1]),
            scale_x * float(intr[2]),
            scale_y * float(intr[3]),
            scale_x * float(intr[4]),
            scale_y * float(intr[5]),
        )

        rotation_yaml = np.asarray(cam1["R"], dtype=np.float64)
        translation_yaml = np.asarray(cam1["t"], dtype=np.float64)
        rotation = rotation_yaml.T
        translation = rotation_yaml @ translation_yaml
        normal = _epipolar_plane(rotation, translation)
        u_axis, v_axis = _plane_basis(normal, translation)

        self.map_x, self.map_y, self.valid_mask = _inverse_map(
            normal,
            u_axis,
            v_axis,
            ds_intrinsics,
            int(output_height),
            int(output_width),
            np.deg2rad(float(half_angle_deg)),
        )
        self.source_width = stream_width
        self.source_height = stream_height
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.half_angle_deg = float(half_angle_deg)

    def rectify(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[:2] != (self.source_height, self.source_width):
            raise ValueError(
                f"Input is {bgr.shape[1]}x{bgr.shape[0]}, expected "
                f"{self.source_width}x{self.source_height} from calibration"
            )
        return cv2.remap(
            bgr,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )


def load_rectifiers(config_path: Path) -> tuple[dict[int, CameraSpec], dict[int, DoubleSphereRectifier], dict[str, Any]]:
    """Load tool config and calibration, resolving paths from the config file."""
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    calibration = Path(config["calibration"]).expanduser()
    if not calibration.is_absolute():
        calibration = (config_path.parent / calibration).resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration YAML not found: {calibration}")
    with calibration.open("r", encoding="utf-8") as stream:
        calibration_data = yaml.safe_load(stream) or {}

    output_width = int(config.get("output_width", 322))
    output_height = int(config.get("output_height", 322))
    half_angle_deg = float(config.get("half_angle_deg", 45.0))
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_width and output_height must be positive")
    if not 0.0 < half_angle_deg < 90.0:
        raise ValueError("half_angle_deg must be between 0 and 90")

    specs: dict[int, CameraSpec] = {}
    rectifiers: dict[int, DoubleSphereRectifier] = {}
    for key, camera in sorted((config.get("cameras") or {}).items(), key=lambda item: int(item[0])):
        camera_id = int(key)
        pair_name = str(camera["calibration_pair"])
        if pair_name not in calibration_data:
            raise KeyError(f"Calibration pair {pair_name!r} is absent from {calibration}")
        spec = CameraSpec(
            camera_id=camera_id,
            input_topic=str(camera.get("input_topic", f"/camera_{camera_id}/image_raw")),
            output_topic=str(camera.get("output_topic", f"/camera_{camera_id}/image_rect")),
            calibration_pair=pair_name,
        )
        specs[camera_id] = spec
        rectifiers[camera_id] = DoubleSphereRectifier(
            calibration_data[pair_name],
            output_height,
            output_width,
            half_angle_deg,
        )

    if not specs:
        raise ValueError("No cameras are configured")

    resolved = dict(config)
    resolved["calibration"] = str(calibration)
    resolved["output_width"] = output_width
    resolved["output_height"] = output_height
    resolved["half_angle_deg"] = half_angle_deg
    resolved["config_source"] = str(config_path)
    return specs, rectifiers, resolved
