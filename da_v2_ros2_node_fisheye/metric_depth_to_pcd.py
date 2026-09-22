"""Convert four calibrated rectangular metric-depth maps to one FRD cloud.

This is the point-cloud helper used by the fisheye ROS inference node.  The
removed standalone ``metric_depth/depth_to_pointcloud.py`` script was not part
of the live ROS point-cloud workflow.
"""

from functools import lru_cache

import numpy as np


# Fallback camera directions for rectified pairs 0_3, 1_0, 2_1, 3_2.
# Calibrated axes supplied by the node take precedence over these values.
# Coordinates follow base_link_frd: +X forward, +Y right, +Z down.
CAMERA_DIRECTIONS = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (-1.0, 0.0, 0.0),
)


@lru_cache(maxsize=8)
def _cached_rays(height, width, stride, intrinsics_key, axes_key):
    """Build all four cameras' pooled-grid rays once per calibrated geometry."""
    pooled_h, pooled_w = height // stride, width // stride
    pixel_u = (np.arange(pooled_w, dtype=np.float32) * stride
               + (stride - 1) / 2.0)
    pixel_v = (np.arange(pooled_h, dtype=np.float32) * stride
               + (stride - 1) / 2.0)
    u, v = np.meshgrid(pixel_u, pixel_v)

    intrinsics = np.asarray(intrinsics_key, np.float32).reshape(-1, 4)
    axes = np.asarray(axes_key, np.float32).reshape(-1, 3, 3)
    fx, fy = intrinsics[:, 0], intrinsics[:, 1]
    cx, cy = intrinsics[:, 2], intrinsics[:, 3]
    x = (u[None] - cx[:, None, None]) / fx[:, None, None]
    y = (v[None] - cy[:, None, None]) / fy[:, None, None]

    forward = axes[:, 0]
    right = axes[:, 1]
    down = axes[:, 2]

    rays = (forward[:, None, None, :]
            + x[..., None] * right[:, None, None, :]
            + y[..., None] * down[:, None, None, :])
    rays = np.ascontiguousarray(rays, dtype=np.float32)
    rays.flags.writeable = False
    return rays


def metric_depth_to_pcd(depth_maps, stride=5, min_depth=0.2,
                        max_depth=20.0, half_angle=np.pi / 4,
                        camera_intrinsics=None, camera_axes=None,
                        camera_origins=None):
    """Return an ``(N, 3)`` float32 XYZ cloud from metric depth in metres.

    Each ``stride x stride`` cell keeps its nearest valid depth. This reduces
    PointCloud2 size without discarding small nearby obstacles. Depth is used
    directly; unlike the old stereo helper, no reciprocal is applied.
    ``camera_axes`` stores [forward, right, down] in base_link_frd and
    ``camera_origins`` stores each optical centre in metres in that frame.
    """
    camera_count = len(CAMERA_DIRECTIONS)
    if len(depth_maps) != camera_count:
        raise ValueError(
            f"Expected {len(CAMERA_DIRECTIONS)} depth maps, got {len(depth_maps)}")
    stride = max(1, int(stride))
    depths = np.asarray(depth_maps, dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(
            f"Expected four equal-size 2-D depth maps, got {depths.shape}")
    _, height, width = depths.shape
    pooled_h, pooled_w = height // stride, width // stride
    if pooled_h == 0 or pooled_w == 0:
        raise ValueError(
            f"stride {stride} is too large for depth shape {height}x{width}")
    used_h, used_w = pooled_h * stride, pooled_w * stride

    if camera_intrinsics is None:
        focal = (width - 1) / (2.0 * np.tan(float(half_angle)))
        camera_intrinsics = np.tile(
            [focal, focal, (width - 1) / 2.0, (height - 1) / 2.0],
            (camera_count, 1))
    camera_intrinsics = np.asarray(camera_intrinsics, np.float32)
    if camera_intrinsics.shape != (camera_count, 4):
        raise ValueError(
            f"camera_intrinsics must be ({camera_count}, 4), got "
            f"{camera_intrinsics.shape}")

    if camera_axes is None:
        forward = np.asarray(CAMERA_DIRECTIONS, dtype=np.float32)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        rig_down = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        down = rig_down[None] - (forward @ rig_down)[:, None] * forward
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        camera_axes = np.stack((forward, right, down), axis=1)
    camera_axes = np.asarray(camera_axes, np.float32)
    if camera_axes.shape != (camera_count, 3, 3):
        raise ValueError(
            f"camera_axes must be ({camera_count}, 3, 3), got "
            f"{camera_axes.shape}")

    if camera_origins is None:
        camera_origins = np.zeros((camera_count, 3), np.float32)
    camera_origins = np.asarray(camera_origins, np.float32)
    if camera_origins.shape != (camera_count, 3):
        raise ValueError(
            f"camera_origins must be ({camera_count}, 3), got "
            f"{camera_origins.shape}")

    valid = (np.isfinite(depths) & (depths >= min_depth)
             & (depths <= max_depth))
    candidates = np.where(valid[:, :used_h, :used_w],
                          depths[:, :used_h, :used_w], np.inf)
    sampled = candidates.reshape(
        camera_count, pooled_h, stride,
        pooled_w, stride).min(axis=(2, 4))
    keep = np.isfinite(sampled)
    if not keep.any():
        return np.empty((0, 3), dtype=np.float32)

    intrinsics_key = tuple(float(v) for v in camera_intrinsics.ravel())
    axes_key = tuple(float(v) for v in camera_axes.ravel())
    rays = _cached_rays(
        height, width, stride, intrinsics_key, axes_key)
    # Avoid NaN/inf multiply warnings in cells that will be discarded below.
    sampled_safe = np.where(keep, sampled, 0.0)
    xyz = (rays * sampled_safe[..., None]
           + camera_origins[:, None, None, :])
    return np.ascontiguousarray(xyz[keep], dtype=np.float32)


def metric_depth_to_colored_pcd(depth_maps, color_maps, stride=5, min_depth=0.2,
                                max_depth=20.0, half_angle=np.pi / 4,
                                camera_intrinsics=None, camera_axes=None,
                                camera_origins=None, bgr_to_rgb=True):
    """Return (xyz, rgb): an ``(N, 3)`` float32 XYZ cloud and an ``(N, 3)`` uint8
    RGB array, aligned index-for-index.

    ``color_maps`` must be four ``(H, W, 3)`` uint8 images in the SAME order and
    SAME frame as ``depth_maps`` — i.e. each pair's LEFT rectified image, since
    the depth is defined in the left rectified camera. Each kept point is colored
    with the exact pixel whose depth won the nearest-depth min-pool, so color and
    geometry come from the same pixel. Set ``bgr_to_rgb=False`` if the images are
    already RGB (the node publishes bgr8, so the default flips).
    """
    camera_count = len(CAMERA_DIRECTIONS)
    if len(depth_maps) != camera_count:
        raise ValueError(
            f"Expected {camera_count} depth maps, got {len(depth_maps)}")
    if len(color_maps) != camera_count:
        raise ValueError(
            f"Expected {camera_count} color maps, got {len(color_maps)}")

    stride = max(1, int(stride))
    depths = np.asarray(depth_maps, dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(
            f"Expected four equal-size 2-D depth maps, got {depths.shape}")
    _, height, width = depths.shape

    colors = np.asarray(color_maps)                       # (C, H, W, 3) uint8
    if colors.shape[:3] != (camera_count, height, width) or colors.shape[3] != 3:
        raise ValueError(
            f"color_maps must be ({camera_count}, {height}, {width}, 3), got "
            f"{colors.shape}")

    pooled_h, pooled_w = height // stride, width // stride
    if pooled_h == 0 or pooled_w == 0:
        raise ValueError(
            f"stride {stride} is too large for depth shape {height}x{width}")
    used_h, used_w = pooled_h * stride, pooled_w * stride

    # ---- intrinsics / axes / origins: identical handling to the mono version --
    if camera_intrinsics is None:
        focal = (width - 1) / (2.0 * np.tan(float(half_angle)))
        camera_intrinsics = np.tile(
            [focal, focal, (width - 1) / 2.0, (height - 1) / 2.0],
            (camera_count, 1))
    camera_intrinsics = np.asarray(camera_intrinsics, np.float32)
    if camera_intrinsics.shape != (camera_count, 4):
        raise ValueError(
            f"camera_intrinsics must be ({camera_count}, 4), got "
            f"{camera_intrinsics.shape}")

    if camera_axes is None:
        forward = np.asarray(CAMERA_DIRECTIONS, dtype=np.float32)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        rig_down = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        down = rig_down[None] - (forward @ rig_down)[:, None] * forward
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        camera_axes = np.stack((forward, right, down), axis=1)
    camera_axes = np.asarray(camera_axes, np.float32)
    if camera_axes.shape != (camera_count, 3, 3):
        raise ValueError(
            f"camera_axes must be ({camera_count}, 3, 3), got {camera_axes.shape}")

    if camera_origins is None:
        camera_origins = np.zeros((camera_count, 3), np.float32)
    camera_origins = np.asarray(camera_origins, np.float32)
    if camera_origins.shape != (camera_count, 3):
        raise ValueError(
            f"camera_origins must be ({camera_count}, 3), got {camera_origins.shape}")

    # ---- nearest-depth pool, but keep the ARGMIN so we can fetch its color ----
    valid = (np.isfinite(depths) & (depths >= min_depth) & (depths <= max_depth))
    candidates = np.where(valid[:, :used_h, :used_w],
                          depths[:, :used_h, :used_w], np.inf)

    # (C, ph, S, pw, S) -> (C, ph, pw, S*S); argmin over the flattened cell
    blocks = (candidates
              .reshape(camera_count, pooled_h, stride, pooled_w, stride)
              .transpose(0, 1, 3, 2, 4)
              .reshape(camera_count, pooled_h, pooled_w, stride * stride))
    flat_idx = blocks.argmin(axis=-1)                     # (C, ph, pw)
    sampled  = np.take_along_axis(blocks, flat_idx[..., None], axis=-1)[..., 0]
    keep = np.isfinite(sampled)                           # (C, ph, pw)
    if not keep.any():
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8))

    # Recover the full-resolution (v, u) of each winning pixel.
    si, sj = np.divmod(flat_idx, stride)                  # offset within cell
    ph = np.arange(pooled_h)[None, :, None]
    pw = np.arange(pooled_w)[None, None, :]
    win_v = ph * stride + si                              # (C, ph, pw)
    win_u = pw * stride + sj
    cam_i = np.arange(camera_count)[:, None, None]
    win_v = np.broadcast_to(win_v, keep.shape)
    win_u = np.broadcast_to(win_u, keep.shape)
    cam_i = np.broadcast_to(cam_i, keep.shape)

    # ---- geometry: cell-center rays (matches the mono function exactly) -------
    intrinsics_key = tuple(float(v) for v in camera_intrinsics.ravel())
    axes_key = tuple(float(v) for v in camera_axes.ravel())
    rays = _cached_rays(height, width, stride, intrinsics_key, axes_key)
    sampled_safe = np.where(keep, sampled, 0.0)
    xyz = rays * sampled_safe[..., None] + camera_origins[:, None, None, :]

    xyz_out = np.ascontiguousarray(xyz[keep], dtype=np.float32)

    # ---- color: gather the winning pixel from each image ----------------------
    rgb_out = colors[cam_i[keep], win_v[keep], win_u[keep]]   # (N, 3) uint8
    if bgr_to_rgb:
        rgb_out = rgb_out[:, ::-1]
    rgb_out = np.ascontiguousarray(rgb_out, dtype=np.uint8)

    return xyz_out, rgb_out
