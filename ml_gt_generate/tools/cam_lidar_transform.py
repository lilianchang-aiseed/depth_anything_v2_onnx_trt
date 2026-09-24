#!/usr/bin/env python3
"""Export one camera-LiDAR calibration JSON for every rectified camera.

The transform composition is intentionally delegated to
``correct_matrix_direction.load_multicamera_lidar_transforms`` so this tool
and the combined-YAML generator always use the same matrix directions and the
same deployed raw-to-rect coordinate maps.

The reference ``calib.json`` supplies:

* ``results.T_lidar_camera`` for one ``/camera_<N>/image_rect`` topic;
* the rectified pinhole intrinsics and distortion copied to every output.

Copying intrinsics is valid only when all rectified topics use the same output
projection.  Each generated JSON records this assumption in ``meta``.

python3 ml_gt_generate/tools/cam_lidar_transform.py \
  --camera-rig calib/2026-09-16/stereo_calib_ds-camchain_ar0234.yaml \
  --reference-calib calib/2026-09-16/cam2-lidar-calib_2_manual/cam2_rect/calib.json \
  --out-dir calib/2026-09-16/cam2-lidar-calib_2_manual \
  --overwrite
  
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

try:  # Package import.
    from .correct_matrix_direction import load_multicamera_lidar_transforms
except ImportError:  # Direct ``python3 tools/cam_lidar_transform.py`` execution.
    from correct_matrix_direction import load_multicamera_lidar_transforms


_RECT_TOPIC = re.compile(r"/camera_(\d+)/image_rect")


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a Hamilton xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {matrix.shape}")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError("rotation determinant is not +1")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0]
                                  - matrix[1, 1] - matrix[2, 2])
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1]
                                  - matrix[0, 0] - matrix[2, 2])
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2]
                                  - matrix[0, 0] - matrix[1, 1])
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # q and -q encode the same rotation. A non-negative w gives deterministic
    # output and matches the convention of the existing calibration JSONs.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def transform_to_vector(transform: np.ndarray) -> list[float]:
    """Return ``[tx, ty, tz, qx, qy, qz, qw]`` for a 4x4 transform."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    quaternion = rotation_to_quaternion_xyzw(transform[:3, :3])
    return [float(value) for value in np.r_[transform[:3, 3], quaternion]]


def export_calibrations(camera_rig: Path, reference_calib: Path,
                        out_dir: Path, overwrite: bool) -> list[Path]:
    camera_rig = camera_rig.expanduser().resolve()
    reference_calib = reference_calib.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    template = json.loads(reference_calib.read_text(encoding="utf-8"))
    reference_topic = str(template.get("meta", {}).get("image_topic", ""))
    match = _RECT_TOPIC.fullmatch(reference_topic)
    if match is None:
        raise ValueError(
            "reference calib meta.image_topic must be "
            f"/camera_<num>/image_rect; got {reference_topic!r}")
    reference_index = int(match.group(1))

    transforms = load_multicamera_lidar_transforms(
        reference_calib, camera_rig)
    if transforms.reference_camera != reference_index:
        raise RuntimeError("reference-camera mismatch after loading calibration")

    outputs = []
    for index, values in transforms.cameras.items():
        output = out_dir / f"cam{index}_rect" / "calib.json"
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output}; pass --overwrite")

        document = deepcopy(template)
        meta = document.setdefault("meta", {})
        meta["image_topic"] = f"/camera_{index}/image_rect"
        meta["derived_from_reference_camera"] = reference_index
        meta["intrinsics_source"] = reference_topic
        meta["camera_rig_calibration"] = str(camera_rig)
        meta["reference_camera_lidar_calibration"] = str(reference_calib)
        results = document.setdefault("results", {})
        vector = transform_to_vector(values["T_lidar_from_rect"])
        results["T_lidar_camera"] = vector
        # A derived camera has no independent optimizer initialization. Keep a
        # structurally compatible field and state its provenance explicitly.
        results["init_T_lidar_camera"] = list(vector)
        meta["init_transform_policy"] = "same_as_derived_final_transform"

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive camera_0..3 rect-to-LiDAR calib.json files from one "
            "reference camera-LiDAR calibration and one four-camera rig YAML"))
    parser.add_argument("--camera-rig", type=Path, required=True,
                        help="four-camera cam_pair_<left>_<right> YAML")
    parser.add_argument("--reference-calib", type=Path, required=True,
                        help="reference /camera_<num>/image_rect calib.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in export_calibrations(
            args.camera_rig, args.reference_calib, args.out_dir,
            args.overwrite):
        print(f"wrote: {output}")


if __name__ == "__main__":
    main()
