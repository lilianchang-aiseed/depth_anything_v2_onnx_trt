#!/usr/bin/env python3
"""ROS 2 multi-pair stereo disparity node for Jetson Orin — batched rectification.

Reads all calibration pairs from ros_stereo/rectify/ds_calib_pairs/:
  stereo_calib_ds_{i}_{j}-camchain.yaml

For each unique camera index a subscription is created.  All pairs sharing a camera
reuse the same message_filters.Subscriber.  ApproximateTimeSynchronizer fires per pair;
a frame collector accumulates decoded images across all pairs and dispatches a complete
4-camera set to a single GPU worker.

GPU worker:
  1. One H2D memcpy: [cam0|cam1|cam2|cam3] concatenated source images
  2. One kernel call: dense_gather_rgb over 2*N_pairs*n_out output cells
  3. One D2H memcpy
  4. N_pairs × TRT inference (sequential) + publish
"""

import collections
import argparse
import glob
import itertools
import os
import queue
import re
import sys
import threading
import types

import numpy as np
import cv2
import yaml
import torch
import torch.nn.functional as F
from easydict import EasyDict

try:
    import cupy as cp
except ImportError:
    cp = None

import rclpy
from rclpy.node import Node
import rclpy.qos
from sensor_msgs.msg import Image, PointCloud2, PointField, LaserScan, Range
import message_filters
from cv_bridge import CvBridge
from inv_depth_to_pcd import disp_to_inv_depth, inv_depth_to_pcd as build_pcd

from datetime import datetime
try:
    from calib_capture import CalibCapture
except ImportError:
    CalibCapture = None

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_RECTIFY_DIR = os.path.join(_HERE, 'rectify')
_OMNI_DIR    = os.path.join(_HERE, 'omniVidar')
_OPENSTEREO  = os.path.join(_OMNI_DIR, 'OpenStereo')
_CALIB_DIR        = os.path.join(_RECTIFY_DIR, 'ds_calib_pairs')
_SINGLE_CALIB_FILE = os.path.join(_RECTIFY_DIR, 'stereo_calib_ds-camchain_from_json.yaml')

sys.path.insert(0, _HERE)
sys.path.insert(0, _RECTIFY_DIR)
sys.path.insert(0, _OPENSTEREO)

# Stub out broken __init__.py files in OpenStereo (h5py / FoundationStereo deps)
for _pkg in ('stereo.modeling', 'stereo.datasets'):
    if _pkg not in sys.modules:
        _parts = _pkg.split('.')
        _m = types.ModuleType(_pkg)
        _m.__path__ = [os.path.join(_OPENSTEREO, *_parts)]
        sys.modules[_pkg] = _m

from distortion_models import unproject_double_sphere_pixels
from rectify_utils import (
    _DENSE_GATHER_KERNEL,
    epiploar_planes_from_extrinsics,
    rasterize_points_map,
    scatter_to_dense_gather,
    rasterize_with_dense_gather,
    shape_to_pixel_grid,
)
from stereo.datasets.dataset_utils.stereo_trans import (
    DivisiblePad, TransposeImage, ToTensor, NormalizeImage, Compose,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
TRT_ENGINE_FILE = os.path.join(_HERE, 'lightstereo_s_batched.engine')

OUTPUT_SIZE    = 320
CONF_THRESHOLD = 0.1
CUT_ANGLE      = np.pi / 4
HALF_ANGLE      = np.pi / 2 - CUT_ANGLE   # ≈ 0.5

_PAIR_RE = re.compile(r'stereo_calib_ds_(\d+)_(\d+)-camchain\.yaml$')

# Stamp quantisation: 50 ms bins match the ApproximateTimeSynchronizer slop so
# that all 4 pair callbacks for the same physical frame map to the same key.
_STAMP_BIN_NS = 50_000_000
RANSAC = True
PER_PIXEL_SCALE = True


# ---------------------------------------------------------------------------
# Helpers (model)
# ---------------------------------------------------------------------------
def _ds_intrinsics(cam_cfg, s=1.):
    intr = cam_cfg['intrinsics']
    return intr[0], intr[1], s*intr[2], s*intr[3], s*intr[4], s*intr[5]


def _build_transform() -> Compose:
    pad_cfg  = EasyDict({'BY': 32, 'MODE': 'tr'})
    norm_cfg = EasyDict({'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225]})
    return Compose([
        DivisiblePad(pad_cfg),
        TransposeImage(EasyDict()),
        ToTensor(EasyDict()),
        NormalizeImage(norm_cfg),
    ])


def add_targeted_pattern(img: torch.Tensor,
                          sigma: float = 0.02,
                          threshold: float = 0.05,
                          freq: float = 20.0) -> torch.Tensor:
    """Add sinusoidal dot pattern to textureless regions to aid stereo matching."""
    _, H, W = img.shape
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
    grad_mag = torch.sqrt(F.conv2d(gray, sobel_x, padding=1) ** 2 +
                          F.conv2d(gray, sobel_y, padding=1) ** 2)
    mask = (grad_mag < threshold).float()
    xs = torch.linspace(0, 2 * torch.pi * freq, W, device=img.device, dtype=img.dtype)
    ys = torch.linspace(0, 2 * torch.pi * freq, H, device=img.device, dtype=img.dtype)
    pattern = (torch.sin(xs).view(1, W) * torch.sin(ys).view(H, 1)).unsqueeze(0)
    return torch.clamp(img + sigma * mask.squeeze(0) * pattern, 0, 1)


class _TRTEngine:
    """TensorRT 10.x inference using PyTorch CUDA tensors."""

    _DTYPE_MAP = {
        np.dtype('float32'): torch.float32,
        np.dtype('float16'): torch.float16,
        np.dtype('int32'):   torch.int32,
        np.dtype('int64'):   torch.int64,
    }

    def __init__(self, engine_path: str):
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self._engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._gpu    = {}
        self._shapes = {}
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            np_dt = trt.nptype(self._engine.get_tensor_dtype(name))
            shape = tuple(self._engine.get_tensor_shape(name))
            th_dt = self._DTYPE_MAP.get(np.dtype(np_dt), torch.float32)
            t = torch.zeros(shape, dtype=th_dt, device='cuda').contiguous()
            self._gpu[name]    = t
            self._shapes[name] = shape
            self._context.set_tensor_address(name, t.data_ptr())

    def __call__(self, left: np.ndarray, right: np.ndarray):
        self._gpu['left_img'].copy_(torch.from_numpy(left),  non_blocking=True)
        self._gpu['right_img'].copy_(torch.from_numpy(right), non_blocking=True)
        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        # Zero out disp_pred where left_img is black (all channels == 0).
        # black_mask = (self._gpu['left_img'].sum(dim=1, keepdim=True) == 0)  # (N,1,H,W)
        # self._gpu['disp_pred'].masked_fill_(black_mask, 0.0)
        disp = self._gpu['disp_pred'].cpu().float().numpy().squeeze()
        # conf = (self._gpu['conf'].cpu().float().numpy().squeeze()
        #         if 'conf' in self._gpu else None)
        conf = None
        return disp, conf


def _infer(engine: _TRTEngine, transform: Compose,
           left_rgb: np.ndarray, right_rgb: np.ndarray):
    """Pre-process → TRT inference → crop."""
    orig_h, orig_w = left_rgb.shape[:2]
    left_t  = torch.from_numpy(left_rgb.astype(np.float32)  / 255.0).permute(2, 0, 1)
    right_t = torch.from_numpy(right_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    left_t  = add_targeted_pattern(left_t)
    right_t = add_targeted_pattern(right_t)
    sample = {
        'left':  left_t.permute(1, 2, 0).numpy() * 255.0,
        'right': right_t.permute(1, 2, 0).numpy() * 255.0,
    }
    sample = transform(sample)
    pad_top, _pr, _pb, pad_left = sample.get('pad', [0, 0, 0, 0])
    left_np  = sample['left'].unsqueeze(0).float().numpy()
    right_np = sample['right'].unsqueeze(0).float().numpy()
    disp, conf = engine(left_np, right_np)
    disp = disp[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]
    if conf is not None:
        conf = conf[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]
    return disp, conf


_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Sobel kernels allocated once at import time (CPU, shared across calls)
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)

# CUDA versions — lazily moved to device on first call to _init_cuda_consts().
# Using None sentinels avoids CUDA context creation at import time.
_CUDA_NORM_MEAN = None
_CUDA_NORM_STD  = None
_CUDA_SOBEL_X   = None
_CUDA_SOBEL_Y   = None
_CUDA_PATTERN   = None   # precomputed (1,1,H,W) sinusoidal pattern on CUDA


def _init_cuda_consts():
    """Move preprocessing constants to CUDA on first call (lazy init)."""
    global _CUDA_NORM_MEAN, _CUDA_NORM_STD, _CUDA_SOBEL_X, _CUDA_SOBEL_Y, _CUDA_PATTERN
    if _CUDA_NORM_MEAN is not None:
        return
    _CUDA_NORM_MEAN = _NORM_MEAN.cuda()
    _CUDA_NORM_STD  = _NORM_STD.cuda()
    _CUDA_SOBEL_X   = _SOBEL_X.cuda()
    _CUDA_SOBEL_Y   = _SOBEL_Y.cuda()
    freq = 20.0
    xs = torch.linspace(0, 2 * torch.pi * freq, OUTPUT_SIZE,
                        device='cuda', dtype=torch.float32)
    ys = torch.linspace(0, 2 * torch.pi * freq, OUTPUT_SIZE,
                        device='cuda', dtype=torch.float32)
    _CUDA_PATTERN = (
        torch.sin(xs).view(1, OUTPUT_SIZE) *
        torch.sin(ys).view(OUTPUT_SIZE, 1)
    ).view(1, 1, OUTPUT_SIZE, OUTPUT_SIZE)


def _infer_batch(engine: _TRTEngine, transform: Compose,
                 left_rgbs: list, right_rgbs: list):
    """Batch TRT inference: process all N pairs in one engine call.

    Vectorised preprocessing — replaces the per-image Python loop in the
    original _infer:
      • np.stack converts all N images in one allocation
      • a single F.conv2d(input=(N,1,H,W)) handles the Sobel pass for all N
        images at once instead of N sequential calls
      • direct (img - mean)/std replaces N transform() pipeline calls
        (valid because OUTPUT_SIZE=320 is always divisible by 32 so
        DivisiblePad is always a no-op and no output crop is needed)

    Args:
        left_rgbs, right_rgbs : lists of N (H, W, 3) uint8 RGB images.
    Returns:
        disp_batch : (N, H, W) float32
        conf_batch : (N, H, W) float32 or None
    """
    sigma, threshold, freq = 0.02, 0.05, 20.0

    # One allocation + one permute for all N images per side
    left_t  = torch.from_numpy(
        np.stack(left_rgbs,  axis=0).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)   # (N, 3, H, W)
    right_t = torch.from_numpy(
        np.stack(right_rgbs, axis=0).astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)

    H, W = left_t.shape[2], left_t.shape[3]
    xs = torch.linspace(0, 2 * torch.pi * freq, W, dtype=torch.float32)
    ys = torch.linspace(0, 2 * torch.pi * freq, H, dtype=torch.float32)
    pattern = (torch.sin(xs).view(1, W) * torch.sin(ys).view(H, 1)).view(1, 1, H, W)

    def _apply_pattern(imgs):
        # imgs: (N, 3, H, W) — one conv2d call handles all N images
        gray = (0.299 * imgs[:, 0] +
                0.587 * imgs[:, 1] +
                0.114 * imgs[:, 2]).unsqueeze(1)           # (N, 1, H, W)
        grad = torch.sqrt(F.conv2d(gray, _SOBEL_X, padding=1) ** 2 +
                          F.conv2d(gray, _SOBEL_Y, padding=1) ** 2)
        mask = (grad < threshold).float()                  # (N, 1, H, W)
        return torch.clamp(imgs + sigma * mask * pattern, 0.0, 1.0)

    left_t  = _apply_pattern(left_t)
    right_t = _apply_pattern(right_t)

    # Normalise to ImageNet stats — equivalent to ToTensor(÷255) + NormalizeImage
    left_np  = ((left_t  - _NORM_MEAN) / _NORM_STD).contiguous().numpy()  # (N,3,H,W)
    right_np = ((right_t - _NORM_MEAN) / _NORM_STD).contiguous().numpy()

    disp_batch, conf_batch = engine(left_np, right_np)
    # 320×320 → DivisiblePad(BY=32) pads nothing; no crop required
    return disp_batch, conf_batch


def _infer_batch_gpu(engine: _TRTEngine, out_all_buf, n_pairs: int,
                     mask: 'torch.Tensor | None' = None,
                     scale: 'torch.Tensor | None' = None,
                     shift: 'torch.Tensor | None' = None):
    """All-GPU batch inference — no D2H→CPU→H2D round-trip.

    Takes the CuPy out_all_buf directly from the rectification kernel,
    performs all preprocessing (BGR→RGB, /255, Sobel pattern, normalize)
    on CUDA via DLPack zero-copy transfer to PyTorch, then feeds the result
    straight into the TRT engine's pre-allocated GPU input buffers.

    Args:
        out_all_buf : CuPy (2*n_pairs*OUTPUT_SIZE²*3,) uint8, still on GPU.
        n_pairs     : number of stereo pairs.
        mask        : (N,64,64) bool CUDA tensor pre-computed at startup;
                      True = valid pixel.  Disparity is zeroed where False.
        scale       : (N,64,64) float32 CUDA tensor; per-pixel multiplier applied
                      as disp = scale * disp + shift after masking.
        shift       : (N,64,64) float32 CUDA tensor; per-pixel additive offset.
    Returns:
        disp_batch  : (n_pairs, H, W) float32 numpy.
        conf_batch  : (n_pairs, H, W) float32 numpy, or None.
    """
    _init_cuda_consts()
    sigma, threshold = 0.02, 0.05

    # Reshape on GPU — no copy, just a metadata change
    rect = out_all_buf.reshape(2 * n_pairs, OUTPUT_SIZE, OUTPUT_SIZE, 3)

    def _cp_bgr_to_cuda_float(cp_bgr):
        """CuPy (N,H,W,3) uint8 BGR → PyTorch CUDA (N,3,H,W) float32.

        Uses zero-copy DLPack handoff; the BGR→RGB channel flip requires
        cp.ascontiguousarray to resolve the negative stride before export.
        """
        rgb = cp.ascontiguousarray(cp_bgr[:, :, :, ::-1])  # (N,H,W,3) uint8 RGB
        t = torch.from_dlpack(rgb.toDlpack())               # (N,H,W,3) uint8 CUDA
        return t.float().div_(255.0).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)

    left_t  = _cp_bgr_to_cuda_float(rect[0::2])
    right_t = _cp_bgr_to_cuda_float(rect[1::2])

    # Black pixel mask: pixels with no rectification output (all channels exactly 0).
    # Computed before texture augmentation while values are still exact zeros.
    # black_mask = (left_t.sum(dim=1, keepdim=True) == 0.0)  # (N, 1, H, W)

    def _apply_pattern_cuda(imgs):
        # imgs: (N, 3, H, W) CUDA float32
        gray = (0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] +
                0.114 * imgs[:, 2]).unsqueeze(1)                    # (N,1,H,W)
        grad = torch.sqrt(
            F.conv2d(gray, _CUDA_SOBEL_X, padding=1) ** 2 +
            F.conv2d(gray, _CUDA_SOBEL_Y, padding=1) ** 2
        )
        mask = (grad < threshold).float()                           # (N,1,H,W)
        return torch.clamp(imgs + sigma * mask * _CUDA_PATTERN, 0.0, 1.0)

    left_t  = _apply_pattern_cuda(left_t)
    right_t = _apply_pattern_cuda(right_t)

    # Normalize (CUDA) — equivalent to NormalizeImage transform
    left_t  = ((left_t  - _CUDA_NORM_MEAN) / _CUDA_NORM_STD).contiguous()
    right_t = ((right_t - _CUDA_NORM_MEAN) / _CUDA_NORM_STD).contiguous()

    # D2D copy into TRT engine's pre-allocated GPU buffers — no H2D needed
    engine._gpu['left_img'].copy_(left_t,  non_blocking=True)
    engine._gpu['right_img'].copy_(right_t, non_blocking=True)
    stream = torch.cuda.current_stream()
    engine._context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Zero out disp_pred where left_img was black (no valid rectification sample).
    # engine._gpu['disp_pred'].masked_fill_(black_mask, 0.0)
    disp = engine._gpu['disp_pred']  # (N, H, W)
    disp = disp.view(4, 64, 5, 64, 5).amin(dim=(2, 4))  # (N, 64, 64)

    # Apply per-pair validity masks: zero out pixels outside the valid FOV.
    if mask is not None:
        disp = disp.masked_fill(~mask, 0.0)

    # Apply per-pixel scale/shift correction: disp = scale * disp + shift.
    if PER_PIXEL_SCALE:
        if scale is not None:
            disp = scale * disp
        if shift is not None:
            disp = disp + shift


    # conf = (engine._gpu['conf'].float().squeeze()
    #         if 'conf' in engine._gpu else None)
    conf = None
    # .reshape(64, 5, 64, 5).max(axis=(1, 3))
    return disp.float().cpu().numpy(), (conf.cpu().numpy() if conf is not None else None)


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------
# Intrinsics of the 64×64 rectified output grid (fx_320 / 5 = 32).
_PCD_FX = _PCD_FY = 32.0
_PCD_CX = _PCD_CY = 31.5


def _inv_depth_64_to_pcd(inv_depths: list, epi_normals: list, shifts: list) -> np.ndarray:
    """Unproject four 64×64 inverse-depth maps to a single (N,3) point cloud.

    Mirrors inv_depth_to_pcd() from inv_depth_to_pcd.py but skips the
    (320→64) max-pool reshape because the GPU worker already downsamples
    to 64×64 on the GPU before returning.

    Args:
        inv_depths  : list of 4 (64,64) float32 arrays (output of disp_to_inv_depth).
        epi_normals : list of 4 length-3 lists/arrays, one per pair.
    Returns:
        (N, 3) float32 XYZ array in the rig frame.
    """
    pts = []
    for inv_d, epi_normal, shift in zip(inv_depths, epi_normals, shifts):
        F_hat = np.asarray(epi_normal, dtype=np.float32)
        F_hat = F_hat / np.linalg.norm(F_hat)

        up    = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # -Z = up in FRD
        D_hat = up - up.dot(F_hat) * F_hat
        D_hat = D_hat / np.linalg.norm(D_hat)
        R_hat = np.cross(D_hat, F_hat)
        R_hat = R_hat / np.linalg.norm(R_hat)

        # inv_d is already 64×64 (GPU max-pool ran on device)
        valid      = (inv_d > 0.05) & (inv_d < 10)
        inv_d_safe = np.where(valid, inv_d, np.nan)
        Z          = 1.0 / inv_d_safe       # depth in metres (NaN where invalid)

        mask    = inv_d > 0.0
        u, v    = np.meshgrid(np.arange(64), np.arange(64))
        F_local = Z[mask]
        R_local = ((u - _PCD_CX) * Z / _PCD_FX)[mask]
        D_local = ((v - _PCD_CY) * Z / _PCD_FY)[mask]

        new_pts = (F_local[:, None] * F_hat
                   + R_local[:, None] * R_hat
                   + D_local[:, None] * D_hat)
        pts.append(new_pts)
    return np.concatenate(pts, axis=0).astype(np.float32)


def pcd_to_laser_scan(pts: np.ndarray) -> np.ndarray:
    """Compute a 72-element laser scan from four 64×64 inverse-depth maps.

    The scan covers 360° in 72 bins of 5° each (bin 0 = [0°,5°), starting from
    the rig +X / forward axis, rotating toward +Y / right in FRD convention).
    Each bin reports the minimum Euclidean distance to any point whose azimuth
    falls within the 5° bin *and* whose elevation from the horizontal plane is
    within ±15° (a 30° vertical cone).

    Args:
        inv_depths  : list of 4 (64,64) float32 arrays (output of disp_to_inv_depth).
        epi_normals : list of 4 length-3 arrays, one per stereo pair.
        shifts      : list of 4 float depth offsets.

    Returns:
        (72,) float32 array of minimum depths in metres; np.inf where no
        valid point falls in a bin.
    """

    scan = np.full(72, np.inf, dtype=np.float32)
    if len(pts) == 0:
        return scan

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Euclidean distance from origin
    dist = np.sqrt(x * x + y * y + z * z)

    # Elevation: atan2(-Z, horiz_range) — Z is down in FRD so -Z is up
    horiz = np.sqrt(x * x + y * y)
    elev_deg = np.degrees(np.arctan2(-z, horiz))

    # Azimuth from +X (forward) rotating toward +Y (right), mapped to [0, 360)
    azimuth_deg = np.degrees(np.arctan2(y, x)) % 360.0

    # 5°-wide bins, index in [0, 71]
    bin_idx = (azimuth_deg / 5.0).astype(np.int32) % 72

    # Keep only points within the ±15° vertical band and with finite depth
    valid = (np.abs(elev_deg) <= 15.0) & np.isfinite(dist)

    np.minimum.at(scan, bin_idx[valid], dist[valid])
    return scan


# Per-pair azimuth bin indices for _inv_depth_64_to_laser_scan_v2.
# Keyed on the epi_normals tuple; populated on first call, reused thereafter.
_ls_v2_bin_cache: dict = {}


def _inv_depth_64_to_laser_scan(inv_depths: list, epi_normals: list, shifts: list) -> tuple:
    """Compute a 72-element laser scan and a floor depth scalar from 64×64
    inverse-depth maps using the same pinhole model as _inv_depth_64_to_pcd
    (fx=fy=32, cx=cy=31.5).

    Step 1 — Squeeze the ±30° vertical band into a 1D column array:
        For pixel (u, v): tan_aR = (u-31.5)/32, tan_aD = (v-31.5)/32.
        The elevation in the rig frame for a horizontal camera is
            elev = atan2(tan_aD, sqrt(1 + tan_aR²))
        so the ±30° band condition becomes column-dependent:
            |tan_aD| ≤ sqrt(1 + tan_aR²) * tan(30°)
        Euclidean range per pixel: depth_fwd * sqrt(1 + tan_aR² + tan_aD²).
        Take the minimum range per column → (64,) array per camera pair.

    Step 2 — Bin columns into the 72-slot scan:
        The azimuth of column u in the rig frame comes from the pinhole ray
            F_hat + tan_aR * R_hat   (depth_fwd cancels out)
        Azimuth = atan2(ray_y, ray_x) % 360°, bin = floor(azim / 5°).
        Accumulate the minimum range per bin across all camera pairs.

    Floor depth — minimum vertical (downward) depth in the 30°–45° band:
        For pixels in the downward 30°–45° band (column-dependent mask on
        tan_aD), the vertical depth is depth_fwd * tan_aD.  The global
        minimum across all such pixels and all camera pairs is returned as a
        single float.

    Args:
        inv_depths  : list of 4 (64,64) float32 arrays.
        epi_normals : list of 4 length-3 arrays, one per stereo pair.
        shifts      : list of 4 float depth offsets.

    Returns:
        scan        : (72,) float32 — horizontal ±30° min ranges (m); inf if empty.
        floor_depth : float — minimum vertical depth in the downward 30°–45°
                      band (m); np.inf if no valid pixel found.
    """
    # Pinhole pixel→tangent grids — computed once, shared across all pairs.
    u_grid, v_grid = np.meshgrid(np.arange(64, dtype=np.float32),
                                 np.arange(64, dtype=np.float32))
    tan_aR_2d = (u_grid - _PCD_CX) / _PCD_FX   # (64,64)
    tan_aD_2d = (v_grid - _PCD_CY) / _PCD_FY   # (64,64)

    # Euclidean range scale: range = depth_fwd * scale
    scale_2d = np.sqrt(1.0 + tan_aR_2d**2 + tan_aD_2d**2)   # (64,64)

    # Column-dependent horizontal scale used for elevation thresholds.
    hor_scale_2d = np.sqrt(1.0 + tan_aR_2d**2)               # (64,64)

    _TAN30 = float(np.tan(np.radians(10.0)))
    _TAN45 = 1.0  # tan(45°) exactly

    # Horizontal band mask: |elev| ≤ 30°.
    vert_mask  = np.abs(tan_aD_2d) <= hor_scale_2d * _TAN30            # (64,64)

    # Floor band mask: downward 30°–45° (positive tan_aD = ray pointing down).
    floor_mask = (tan_aD_2d >= hor_scale_2d * _TAN30) & \
                 (tan_aD_2d <= hor_scale_2d * _TAN45)                  # (64,64)

    # 1D column tangents for azimuth mapping (tan_aR is constant along rows).
    tan_aR_1d = tan_aR_2d[0, :]   # (64,)

    # Fetch or build cached per-pair bin indices.
    # bin_idx depends only on epi_normals (camera calibration), not on depth data,
    # so it is computed once and reused across every subsequent call.
    cache_key = tuple(tuple(float(x) for x in n) for n in epi_normals)
    if cache_key not in _ls_v2_bin_cache:
        bin_idx_list = []
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        for epi_normal in epi_normals:
            F_hat = np.asarray(epi_normal, dtype=np.float32)
            F_hat = F_hat / np.linalg.norm(F_hat)
            D_hat = up - up.dot(F_hat) * F_hat
            D_hat = D_hat / np.linalg.norm(D_hat)
            R_hat = np.cross(D_hat, F_hat)
            R_hat = R_hat / np.linalg.norm(R_hat)
            ray_x = F_hat[0] + tan_aR_1d * R_hat[0]
            ray_y = F_hat[1] + tan_aR_1d * R_hat[1]
            azim_deg = np.degrees(np.arctan2(ray_y, ray_x)) % 360.0
            bin_idx_list.append((azim_deg / 5.0).astype(np.int32) % 72)
        _ls_v2_bin_cache[cache_key] = bin_idx_list
    bin_idx_list = _ls_v2_bin_cache[cache_key]

    scan        = np.full(72, np.inf, dtype=np.float32)
    floor_depth = np.inf

    # Hot loop: only depth-dependent work; bin_idx comes from the cache.
    for inv_d, shift, bin_idx in zip(inv_depths, shifts, bin_idx_list):
        valid_inv = (inv_d > 0.05) & (inv_d < 0.2)
        depth_fwd = np.where(valid_inv, 1.0 / inv_d, np.nan)

        # Horizontal scan: ±30° band, min Euclidean range per column → azimuth bins.
        band_valid    = valid_inv & vert_mask                              # (64,64)
        range_map     = np.where(band_valid, depth_fwd * scale_2d, np.inf)
        min_range_col = range_map.min(axis=0)                              # (64,)
        good = np.isfinite(min_range_col) & (min_range_col > 0.0)
        np.minimum.at(scan, bin_idx[good], min_range_col[good].astype(np.float32))

        # Floor depth: downward 30°–45° band, vertical component = depth_fwd * tan_aD.
        floor_valid  = valid_inv & floor_mask                              # (64,64)
        vert_depth   = np.where(floor_valid, depth_fwd * tan_aD_2d, np.inf)
        pair_min     = float(vert_depth.min())
        if pair_min < floor_depth:
            floor_depth = pair_min

    return scan, floor_depth


def _make_pcd_msg(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Pack (N, 3) float32 XYZ array into a PointCloud2 message."""
    pts = pts.astype(np.float32)
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height   = 1
    msg.width    = len(pts)
    msg.fields   = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(pts)
    msg.data         = pts.tobytes()
    msg.is_dense     = False
    return msg

def _make_laser_msg(laser: np.ndarray, stamp, frame_id: str) -> LaserScan:
    """Pack a 72-element range array into a LaserScan message.

    The scan covers 360° in 72 bins of 5° each, starting at 0° (rig +X / forward).
    Bins with no valid return carry np.inf, which is the standard ROS convention
    for 'no obstacle detected'.
    """
    _ANGLE_INC = float(np.radians(5.0))   # 5° per bin
    msg = LaserScan()
    msg.header.stamp     = stamp
    msg.header.frame_id  = frame_id
    msg.angle_min        = 0.0
    msg.angle_max        = 71 * _ANGLE_INC          # angle of the last beam
    msg.angle_increment  = _ANGLE_INC
    msg.time_increment   = 0.0
    msg.scan_time        = 0.0
    msg.range_min        = 0.1
    msg.range_max        = 20.0
    msg.ranges           = laser.astype(np.float32).tolist()
    msg.intensities      = []
    return msg


def _make_debug_laser_msg(stamp, frame_id: str = 'base_link_frd') -> LaserScan:
    """Pack a constant 72-element debug range scan for MAVROS obstacle testing."""
    laser = np.ones(72, dtype=np.float32) * 10.0
    laser[33:38] = 5
    _ANGLE_INC = float(np.radians(5.0))
    msg = LaserScan()
    msg.header.stamp     = stamp
    msg.header.frame_id  = frame_id
    msg.angle_min        = 0.0
    msg.angle_max        = 71 * _ANGLE_INC
    msg.angle_increment  = _ANGLE_INC
    msg.time_increment   = 0.0
    msg.scan_time        = 0.0
    msg.range_min        = 0.1
    msg.range_max        = 100.0
    msg.ranges           = laser.tolist()
    msg.intensities      = []
    return msg


def _make_terrain_msg(floor_depth: float, stamp, frame_id: str = 'base_link_frd') -> Range:
    """Pack the minimum floor vertical depth into a Range message for MAVROS.

    Published to /distance_sensor/terrain_estimator.  MAVROS plugin config
    should set orientation=PITCH_270, field_of_view=2.0, send_tf=false.
    """
    msg = Range()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.radiation_type  = Range.INFRARED
    msg.field_of_view   = 2.0
    msg.min_range       = 0.1
    msg.max_range       = 20.0
    msg.range           = float(np.clip(floor_depth, 0.1, 20.0)) \
                          if np.isfinite(floor_depth) else float('inf')
    return msg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class MultiStereoDisparityNode(Node):
    """Run stereo disparity for all calibrated camera pairs in ds_calib_pairs/."""

    #def __init__(self, obstacle_debug: bool = False):
    def __init__(self, obstacle_debug: bool = False, calib_args: dict = None):
        super().__init__('multi_stereo_disparity')
        self._bridge = CvBridge()
        self._obstacle_debug = obstacle_debug
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._amp    = (self._device.type == 'cuda')

        self._stereo_calib = self._load_stereo_calib()
        self._pairs = self._discover_pairs()
        if not self._pairs:
            self.get_logger().error(f'No pairs found in {_SINGLE_CALIB_FILE}')
            return

        self.get_logger().info(
            f'Found {len(self._pairs)} stereo pair(s): '
            + ', '.join(f'({i},{j})' for i, j, _ in self._pairs))

        # Unique camera indices in sorted order (e.g. [0, 1, 2, 3])
        self._cam_indices = sorted({idx for p in self._pairs for idx in (p[0], p[1])})
        self._n_cams = len(self._cam_indices)

        # GPU worker: ROS callbacks only decode+collect; worker handles all GPU work
        self._stop_evt     = threading.Event()
        self._work_queue   = queue.Queue(maxsize=1)
        # Per-camera pools: each holds the 2 most recent decoded frames.
        # (timestamp_ns: int, bgr: np.ndarray, stamp_msg: Header.stamp)
        self._pool_lock       = threading.Lock()
        self._cam_pools       = {c: collections.deque(maxlen=2) for c in self._cam_indices}
        self._last_dispatch_ns = 0   # median ns of last dispatched frame set
        self._gpu_thread   = threading.Thread(target=self._gpu_worker, daemon=True)
        self._gpu_thread.start()

        # Build per-pair dense gather tables
        self.get_logger().info('Pre-computing rectification maps...')
        self._rect_maps = {}
        for left_idx, right_idx, calib_data in self._pairs:
            self.get_logger().info(
                f'  Pair ({left_idx},{right_idx}): cam_pair_{left_idx}_{right_idx}')
            self._rect_maps[(left_idx, right_idx)] = self._build_rect_maps(calib_data)

        # Merge all tables into one mega-table for a single batch kernel call
        self._build_batch_tables()

        # Load shared LightStereo TRT model
        self.get_logger().info('Loading LightStereo model...')
        self._build_model()

        # Precompute per-pair per-column azimuth-bin scale/shift corrections.
        self._build_bin_corrections()

        # Pre-load per-pair validity masks, scale, and shift onto GPU (done once; pairs are fixed).
        # mask  shape: (N, 64, 64) bool,    True = valid disparity pixel.
        # scale shape: (N, 64, 64) float32, per-pixel multiplicative correction.
        # shift shape: (N, 64, 64) float32, per-pixel additive correction.
        _masks_dir = os.path.join(_RECTIFY_DIR, 'masks')
        self._combined_mask = torch.stack([
            torch.from_numpy(
                np.load(os.path.join(_masks_dir, f'pair_{i}_{j}.npy')))
            for i, j, _ in self._pairs
        ], dim=0).cuda()
        self._combined_scale = torch.stack([
            torch.from_numpy(
                np.load(os.path.join(_masks_dir, f'scale_{i}_{j}.npy')).astype(np.float32))
            for i, j, _ in self._pairs
        ], dim=0).cuda()
        self._combined_shift = torch.stack([
            torch.from_numpy(
                np.load(os.path.join(_masks_dir, f'shift_{i}_{j}.npy')).astype(np.float32))
            for i, j, _ in self._pairs
        ], dim=0).cuda()

        # Publishers + one subscriber per unique camera (shared across syncs)
        self._pub_left  = {}
        self._pub_right = {}
        self._pub_disp  = {}
        self._pub_pcd   = self.create_publisher(PointCloud2, '/stereo/point_cloud', 1)
        self._pub_laser   = self.create_publisher(LaserScan, '/mavros/obstacle/send', 1)
        self._pub_terrain = self.create_publisher(Range, '/distance_sensor/terrain_estimator', 1)
        if self._obstacle_debug:
            self.create_timer(0.1, self._publish_obstacle_debug)
            self.get_logger().info('Obstacle debug enabled: publishing /mavros/obstacle/send every 100 ms')
        # One direct subscription per unique camera; pool-based sync replaces ATS.
        _qos = rclpy.qos.QoSProfile(depth=4)
        self._cam_subs = []
        for cam_idx in self._cam_indices:
            sub = self.create_subscription(
                Image,
                f'/camera_{cam_idx}/image_raw',
                lambda msg, idx=cam_idx: self._cam_callback(idx, msg),
                qos_profile=_qos)
            self._cam_subs.append(sub)

        for left_idx, right_idx, _ in self._pairs:
            ns = f'/stereo_{left_idx}_{right_idx}'
            self._pub_left[(left_idx, right_idx)]  = self.create_publisher(
                Image, f'{ns}/left/image_rect',  1)
            self._pub_right[(left_idx, right_idx)] = self.create_publisher(
                Image, f'{ns}/right/image_rect', 1)
            self._pub_disp[(left_idx, right_idx)]  = self.create_publisher(
                Image, f'{ns}/disparity',        1)

            self.get_logger().info(
                f'  Pair ({left_idx},{right_idx}): '
                f'/camera_{left_idx} + /camera_{right_idx} → {ns}/')

        self.get_logger().info(
            f'Ready — device={self._device}, AMP={self._amp}, '
            f'output_size={OUTPUT_SIZE}, pairs={len(self._pairs)}, '
            f'cameras={self._n_cams}')

        # ---- optional calibration-set capture ----------------------------
        self._calib = None
        self._calib_pair_pi = None
        if calib_args and calib_args.get('enabled'):
            if CalibCapture is None:
                self.get_logger().error(
                    'calib requested but calib_capture.py not importable')
            else:
                lp, rp = calib_args['pair']
                for pi, (li, ri, _) in enumerate(self._pairs):
                    if (li, ri) == (lp, rp):
                        self._calib_pair_pi = pi
                        break
                if self._calib_pair_pi is None:
                    self.get_logger().error(
                        f'calib pair {lp}_{rp} not among discovered pairs '
                        f'— calib disabled')
                else:
                    self._calib = CalibCapture(
                        self, calib_args,
                        left_topic=f'/stereo_{lp}_{rp}/left/image_rect',
                        right_topic=f'/stereo_{lp}_{rp}/right/image_rect')
                    self.get_logger().info(
                        f'Calib capture ON: pair {lp}_{rp} '
                        f'(pi={self._calib_pair_pi})')

    def _publish_obstacle_debug(self):
        stamp = self.get_clock().now().to_msg()
        debug_msg = _make_debug_laser_msg(stamp, 'base_link_frd')
        self._pub_laser.publish(debug_msg)

    # ------------------------------------------------------------------
    def _load_stereo_calib(self) -> dict:
        """Load baseline / focal-length per pair from stereo_rectified.yaml.

        Returns a dict keyed by '{left}_{right}' with entries
        {'baseline_m': float, 'fx_px': float}.
        """
        calib_file = os.path.join(_RECTIFY_DIR, 'stereo_rectified.yaml')
        with open(calib_file) as f:
            raw = yaml.safe_load(f)
        calib = {}
        for key, val in raw.items():
            calib[key] = {
                'baseline_m':   val['baseline_m'],
                'f_shift':      val['f_shift'],
                'fx_px':        val['fx_rect_px'],
                'epipolar_norm': val['epipolar_norm'],
                'ransac_scale': val['ransac_scale'],
                'scale': 1 / (val['baseline_m']*val['fx_rect_px']),
                'ransac_shift': val['ransac_shift'],
            }
        self.get_logger().info(
            f'Loaded stereo calib for pairs: {list(calib.keys())}')
        return calib

    # ------------------------------------------------------------------
    def _discover_pairs(self):
        _pair_re = re.compile(r'^cam_pair_(\d+)_(\d+)$')
        with open(_SINGLE_CALIB_FILE) as f:
            all_calib = yaml.safe_load(f)
        pairs = []
        for key in sorted(all_calib.keys()):
            m = _pair_re.match(key)
            if m:
                pairs.append((int(m.group(1)), int(m.group(2)), all_calib[key]))
        return pairs

    # ------------------------------------------------------------------
    def _build_rect_maps(self, calib: dict):
        """Pre-compute dense neighbor tables for both cameras from a calib dict.

        The dict has cam0/cam1 entries with intrinsics, resolution, and
        separate R (3×3) and t (3,) fields in place of T_cn_cnm1.

        Returns (tables0, tables1), each a dict from scatter_to_dense_gather.
        """
        cam0_cfg = calib['cam0']
        cam1_cfg = calib['cam1']

        cW, cH = cam0_cfg['resolution']
        W, H = calib.get('stream_resolution', (cW, cH))
        scale_calib = H / cH
        image_shape = (H, W)
        self.get_logger().info(f'    Calibration image shape: {image_shape}')

        pixels = shape_to_pixel_grid(image_shape, stride=1)

        self.get_logger().info('    Unprojecting cam0 rays...')
        xi, alpha, fx, fy, cx, cy = _ds_intrinsics(cam0_cfg, scale_calib)
        rays0, _ = unproject_double_sphere_pixels(pixels, xi, alpha, fx, fy, cx, cy)

        self.get_logger().info('    Unprojecting cam1 rays...')
        xi, alpha, fx, fy, cx, cy = _ds_intrinsics(cam1_cfg, scale_calib)
        rays1, _ = unproject_double_sphere_pixels(pixels, xi, alpha, fx, fy, cx, cy)

        # R and t are stored separately (decoupled from T_cn_cnm1)
        R_mat = np.array(cam1_cfg['R'], dtype=np.float32)
        t_vec = np.array(cam1_cfg['t'], dtype=np.float32)
        R = R_mat.T
        t = R_mat @ t_vec

        n0, n1 = epiploar_planes_from_extrinsics(R, t)
        n0 = n0.astype(np.float32)
        n1 = n1.astype(np.float32)

        self.get_logger().info('    Building rasterisation maps...')
        map0, coord = rasterize_points_map(
            n0, rays0.astype(np.float32), t, n=OUTPUT_SIZE, b2n=HALF_ANGLE)
        cam1_coord = [R.T @ e for e in coord]
        map1, _ = rasterize_points_map(
            n1, rays1.astype(np.float32), t, n=OUTPUT_SIZE, b2n=HALF_ANGLE,
            coord=cam1_coord)

        self.get_logger().info('    Building dense gather tables...')
        tables0 = scatter_to_dense_gather(map0, OUTPUT_SIZE)
        tables1 = scatter_to_dense_gather(map1, OUTPUT_SIZE)
        self.get_logger().info(
            f'    MAX_K={tables0["MAX_K"]} contributors/cell, '
            f'idx table: {tables0["MAX_K"]}×{tables0["n_out"]}')

        self.get_logger().info('    Rectification maps ready.')
        return tables0, tables1

    # ------------------------------------------------------------------
    def _build_batch_tables(self):
        """Merge all per-pair dense gather tables into one mega-table.

        Assigns each unique camera a slot in a concatenated source buffer:
            src_all = [cam_indices[0] | cam_indices[1] | ...]
        The slot offset (cam_slot × N_source) is baked into each gather_idx table
        at startup so the kernel addresses all cameras via one flat src pointer.

        Output layout in out_all_buf — for each pair in self._pairs order,
        left rectification immediately followed by right rectification:
            [pair0_left | pair0_right | pair1_left | pair1_right | ...]
        Reshaped as (2 × N_pairs, OUTPUT_SIZE, OUTPUT_SIZE, 3) after D2H.

        Falls back gracefully if CuPy is unavailable (per-pair sequential path).
        """
        if cp is None:
            self.get_logger().warn(
                'CuPy not available — batch mode disabled; per-pair fallback active')
            self._batch_idx = None
            return

        cam_slot = {c: k for k, c in enumerate(self._cam_indices)}

        # Geometry constants are identical across all tables (same OUTPUT_SIZE)
        first_tables = next(iter(self._rect_maps.values()))
        N_source  = first_tables[0]['N_source']
        n_out     = first_tables[0]['n_out']
        self._n_out = n_out

        # Use global MAX_K so all tables can be concatenated; pad shorter tables
        global_max_k = max(
            t['MAX_K']
            for pair_tables in self._rect_maps.values()
            for t in pair_tables
        )

        idx_list = []
        w_list   = []

        for left_idx, right_idx, _ in self._pairs:
            tables0, tables1 = self._rect_maps[(left_idx, right_idx)]

            for cam_idx, t in ((left_idx, tables0), (right_idx, tables1)):
                # Bake source-buffer slot offset into every index.
                # Unused slots have w=0 so the dummy load at cam_slot*N_source
                # contributes nothing and stays within src_all_buf bounds.
                idx = t['idx'] + cam_slot[cam_idx] * N_source

                cur_k = idx.shape[0]
                if cur_k < global_max_k:
                    pad_k = global_max_k - cur_k
                    zero  = cp.zeros((pad_k, n_out), dtype=idx.dtype)
                    idx = cp.concatenate([idx, zero],                          axis=0)
                    w   = cp.concatenate([t['w'],
                                          cp.zeros((pad_k, n_out),
                                                   dtype=t['w'].dtype)],       axis=0)
                else:
                    w = t['w']

                idx_list.append(idx)
                w_list.append(w)

        # Concatenate along n_out axis → shape (MAX_K, 2*N_pairs*n_out)
        self._batch_max_k = global_max_k
        self._batch_idx   = cp.concatenate(idx_list, axis=1)
        self._batch_w     = cp.concatenate(w_list,   axis=1)

        # Pre-allocated GPU I/O (no per-frame cudaMalloc)
        self._src_all_buf = cp.empty(self._n_cams * N_source * 3, dtype=cp.uint8)
        self._out_all_buf = cp.zeros(2 * len(self._pairs) * n_out * 3, dtype=cp.uint8)

        self.get_logger().info(
            f'Batch table: MAX_K={global_max_k}, '
            f'shape={tuple(self._batch_idx.shape)}, '
            f'src_buf={self._n_cams}×{N_source}×3 B, '
            f'out_buf={2*len(self._pairs)}×{n_out}×3 B')

    # ------------------------------------------------------------------
    def _build_model(self):
        self.get_logger().info(f'  Loading TRT engine: {TRT_ENGINE_FILE}')
        self._trt       = _TRTEngine(TRT_ENGINE_FILE)
        self._transform = _build_transform()
        self.get_logger().info('  TensorRT engine loaded.')

    # ------------------------------------------------------------------
    def _build_bin_corrections(self):
        """Precompute per-pair per-column azimuth-bin scale/shift (72-bin arrays).

        For each pair, maps each of the 64 output columns to a laser-scan
        azimuth bin (0–71) using the same pinhole + epipolar geometry as
        _inv_depth_64_to_laser_scan.  The (72,) scale.npy / shift.npy
        calibration arrays are indexed by that bin to produce per-column
        (64,) correction vectors stored as self._col_scale / self._col_shift.

        NaN entries in scale.npy / shift.npy (uncalibrated bins) default to
        scale=1, shift=0 so the original inv_depth passes through unchanged.
        """
        scale_arr = np.load(os.path.join(_RECTIFY_DIR, 'scale.npy')).astype(np.float32)
        shift_arr = np.load(os.path.join(_RECTIFY_DIR, 'shift.npy')).astype(np.float32)

        tan_aR_1d = (np.arange(64, dtype=np.float32) - _PCD_CX) / _PCD_FX  # (64,)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self._col_scale = []
        self._col_shift = []
        self._col_nan   = []   # (64,) bool: True = uncalibrated bin → output 100
        for left_idx, right_idx, _ in self._pairs:
            calib = self._stereo_calib[f'{left_idx}_{right_idx}']
            F_hat = np.asarray(calib['epipolar_norm'], dtype=np.float32)
            F_hat = F_hat / np.linalg.norm(F_hat)
            D_hat = up - up.dot(F_hat) * F_hat
            D_hat = D_hat / np.linalg.norm(D_hat)
            R_hat = np.cross(D_hat, F_hat)
            R_hat = R_hat / np.linalg.norm(R_hat)

            ray_x    = F_hat[0] + tan_aR_1d * R_hat[0]
            ray_y    = F_hat[1] + tan_aR_1d * R_hat[1]
            azim_deg = np.degrees(np.arctan2(ray_y, ray_x)) % 360.0
            bin_idx  = (azim_deg / 5.0).astype(np.int32) % 72  # (64,)

            raw_scale = scale_arr[bin_idx]
            raw_shift = shift_arr[bin_idx]
            nan_mask  = np.isnan(raw_scale)                      # (64,) bool
            col_scale = np.where(nan_mask, 1.0, raw_scale)
            col_shift = np.where(nan_mask, 0.0, raw_shift)
            self._col_scale.append(col_scale)
            self._col_shift.append(col_shift)
            self._col_nan.append(nan_mask)

        self.get_logger().info(
            f'Bin corrections loaded: {len(self._col_scale)} pairs, '
            f'scale range [{np.nanmin(scale_arr):.3f}, {np.nanmax(scale_arr):.3f}]')

    # ------------------------------------------------------------------
    def _cam_callback(self, cam_idx: int, msg: Image):
        """Decode one camera frame, add to its pool, then try to assemble a complete set."""
        bgr      = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        with self._pool_lock:
            self._cam_pools[cam_idx].append((stamp_ns, bgr, msg.header.stamp))
            entry = self._try_assemble()

        if entry is None:
            return

        try:
            self._work_queue.put_nowait(entry)
        except queue.Full:
            try:
                self._work_queue.get_nowait()   # evict oldest, keep latest
            except queue.Empty:
                pass
            self._work_queue.put_nowait(entry)
            self.get_logger().warn(
                'Older frame dropped, prioritizing latest (GPU worker behind)',
                throttle_duration_sec=1.0)

    # ------------------------------------------------------------------
    def _try_assemble(self):
        """Select one frame per camera so the timestamp spread is minimised.

        Called with _pool_lock held.  Returns a work-queue entry dict or None.
        Tries all 2^n combinations (n ≤ 4 cameras, pool depth 2 → ≤ 16 combos).
        """
        pools = [list(self._cam_pools[c]) for c in self._cam_indices]
        if any(len(p) == 0 for p in pools):
            return None   # at least one camera has no frames yet

        best_combo  = None
        best_spread = float('inf')
        for combo in itertools.product(*pools):
            stamps = [item[0] for item in combo]
            spread = max(stamps) - min(stamps)
            if spread < best_spread:
                best_spread = spread
                best_combo  = combo

        # Canonical timestamp: median of selected frames (avoids re-dispatch of same set)
        sorted_ns = sorted(item[0] for item in best_combo)
        median_ns = sorted_ns[len(sorted_ns) // 2]

        if median_ns <= self._last_dispatch_ns:
            return None   # already dispatched this set or an older one

        self._last_dispatch_ns = median_ns
        return {
            'images':   {self._cam_indices[i]: best_combo[i][1]
                         for i in range(self._n_cams)},
            'stamps':   {self._cam_indices[i]: best_combo[i][2]
                         for i in range(self._n_cams)},   # <-- add
            'stamp':    best_combo[0][2],
            'frame_id': 'base_link_frd',
        }

    # ------------------------------------------------------------------
    def _gpu_worker(self):
        """Background thread: batch rect → N_pairs × TRT → publish."""
        _t_last_done = None
        while not self._stop_evt.is_set():
            try:
                entry = self._work_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            _t_got = self.get_clock().now()
            # True idle time = gap from end of last frame to receipt of this one.
            # The old _t_wait0/1 approach reset on every timeout, hiding multi-timeout waits.
            wait_ms = (_t_got - _t_last_done).nanoseconds / 1e6 if _t_last_done is not None else 0.0

            images   = entry['images']
            stamps   = entry['stamps']
            stamp    = entry['stamp']
            frame_id = 'base_link_frd'
            t0 = self.get_clock().now()

            # ---- Batch rectification: 1 H2D + 1 kernel -----------------------
            if cp is not None and self._batch_idx is not None:
                # Assemble source buffer in camera-slot order
                src_all = np.concatenate(
                    [images[c].ravel() for c in self._cam_indices]
                )
                self._src_all_buf.set(src_all)   # H2D, pre-allocated — no malloc

                n_total = 2 * len(self._pairs) * self._n_out
                blocks  = (n_total + 255) // 256

                _DENSE_GATHER_KERNEL(
                    (blocks,), (256,),
                    (self._src_all_buf, self._batch_idx, self._batch_w,
                     np.int32(self._batch_max_k), np.int32(n_total),
                     self._out_all_buf),
                )
                # D2H deferred — out_all_buf passed directly to GPU infer below
            else:
                # CPU / no-CuPy fallback: sequential per-pair
                rects = []
                for left_idx, right_idx, _ in self._pairs:
                    tables0, tables1 = self._rect_maps[(left_idx, right_idx)]
                    rects.append(
                        rasterize_with_dense_gather(
                            tables0, images[left_idx],  OUTPUT_SIZE))
                    rects.append(
                        rasterize_with_dense_gather(
                            tables1, images[right_idx], OUTPUT_SIZE))
                rect_all = np.stack(rects)

            t1 = self.get_clock().now()
            rect_ms = (t1 - t0).nanoseconds / 1e6

            # ---- Batch TRT inference: all N_pairs in one engine call ----------
            if cp is not None and self._batch_idx is not None:
                # All-GPU path: BGR→RGB + Sobel + normalize all on CUDA,
                # then feed directly into TRT GPU buffers — no D2H/H2D needed.
                disp_batch, conf_batch = _infer_batch_gpu(
                    self._trt, self._out_all_buf, len(self._pairs),
                    self._combined_mask, self._combined_scale, self._combined_shift)
                # D2H for publishing happens AFTER TRT (rect kernel output still valid)
                _t_d2h0 = self.get_clock().now()
                rect_all = (self._out_all_buf.get()
                            .reshape(2 * len(self._pairs),
                                     OUTPUT_SIZE, OUTPUT_SIZE, 3))
                _t_d2h1 = self.get_clock().now()
            else:
                left_rgbs  = [cv2.cvtColor(rect_all[pi * 2    ], cv2.COLOR_BGR2RGB)
                              for pi in range(len(self._pairs))]
                right_rgbs = [cv2.cvtColor(rect_all[pi * 2 + 1], cv2.COLOR_BGR2RGB)
                              for pi in range(len(self._pairs))]
                disp_batch, conf_batch = _infer_batch(
                    self._trt, self._transform, left_rgbs, right_rgbs)

            t2 = self.get_clock().now()
            infer_ms = (t2 - t1).nanoseconds / 1e6
            d2h_ms = (_t_d2h1 - _t_d2h0).nanoseconds / 1e6 if cp is not None and self._batch_idx is not None else 0.0

            # ---- Point cloud from 64×64 disparity ---------------------------
            _t_prep0 = self.get_clock().now()
            inv_depths  = []
            epi_normals = []
            shifts = []
            for pi, (left_idx, right_idx, _) in enumerate(self._pairs):
                calib = self._stereo_calib[f'{left_idx}_{right_idx}']
                shift = calib['ransac_shift'] if RANSAC else calib['f_shift']
                scale = calib['ransac_scale'] if RANSAC else calib['scale']
                inv_depths.append(disp_to_inv_depth(disp_batch[pi], scale, shift))
                epi_normals.append(calib['epipolar_norm'])
                
                shifts.append(shift)
            # for pi in range(len(inv_depths)):
            #     s  = self._col_scale[pi][np.newaxis, :]   # (1, 64) → broadcast rows
            #     sh = self._col_shift[pi][np.newaxis, :]   # (1, 64)
            #     inv_depths[pi] = s * inv_depths[pi] + sh
            #     inv_depths[pi][:, self._col_nan[pi]] = 100.0
            _t_prep1 = self.get_clock().now()
            prep_ms = (_t_prep1 - _t_prep0).nanoseconds / 1e6

            t1 = self.get_clock().now()
            pcd_pts = _inv_depth_64_to_pcd(inv_depths, epi_normals, shifts)
            # laser = pcd_to_laser_scan(pcd_pts)
            
            # laser, floor_depth = _inv_depth_64_to_laser_scan(inv_depths, epi_normals, shifts)
            t2 = self.get_clock().now()
            pcd_ms = (t2 - t1).nanoseconds / 1e6
            pcd_msg = _make_pcd_msg(pcd_pts, stamp, frame_id)
            self._pub_pcd.publish(pcd_msg)
            _t_pub0 = self.get_clock().now()
            # if not self._obstacle_debug:
            #     laser_msg = _make_laser_msg(laser, stamp, frame_id)
            #     self._pub_laser.publish(laser_msg)
            # terrain_msg = _make_terrain_msg(floor_depth, stamp)
            # self._pub_terrain.publish(terrain_msg)
            _t_pub1 = self.get_clock().now()
            pub_ms = (_t_pub1 - _t_pub0).nanoseconds / 1e6

            _t_last_done = self.get_clock().now()
            total_ms = wait_ms + rect_ms + infer_ms + d2h_ms + prep_ms + pcd_ms + pub_ms
            self.get_logger().info(
                f'wait {wait_ms:.0f} ms | rect {rect_ms:.0f} ms | infer {infer_ms:.0f} ms'
                f' | d2h {d2h_ms:.0f} ms | prep {prep_ms:.0f} ms | pcd {pcd_ms:.0f} ms'
                f' | pub {pub_ms:.0f} ms | TOTAL {total_ms:.0f} ms',
                throttle_duration_sec=1.0)
            
            # ---- Images Publish ------------------------------------------------------
            for pi, (left_idx, right_idx, _) in enumerate(self._pairs):
                # disp = inv_depths[pi]
                disp = disp_batch[pi]
                # conf = conf_batch[pi] if conf_batch is not None else None
                # if conf is not None:
                #     disp = np.where(conf > CONF_THRESHOLD, disp, 0.0).astype(np.float32)

                rect_left  = rect_all[pi * 2    ]
                rect_right = rect_all[pi * 2 + 1]

                pair_stamp = stamps[left_idx]

                left_out = self._bridge.cv2_to_imgmsg(rect_left,  encoding='bgr8')
                #left_out.header.stamp    = stamp
                left_out.header.stamp    = pair_stamp
                left_out.header.frame_id = frame_id
                self._pub_left[(left_idx, right_idx)].publish(left_out)

                right_out = self._bridge.cv2_to_imgmsg(rect_right, encoding='bgr8')
                #right_out.header.stamp    = stamp
                right_out.header.stamp    = pair_stamp
                right_out.header.frame_id = frame_id
                self._pub_right[(left_idx, right_idx)].publish(right_out)

                # valid_inv = (disp > 0.05) & (disp < 2)
                # disp = np.where(valid_inv, 1.0 / disp + shift, np.nan)

                disp_out = self._bridge.cv2_to_imgmsg(disp, encoding='32FC1')
                #disp_out.header.stamp    = stamp
                disp_out.header.stamp    = pair_stamp
                disp_out.header.frame_id = frame_id
                self._pub_disp[(left_idx, right_idx)].publish(disp_out)

            # ---- calibration triplet capture --------------------------------
            if self._calib is not None and self._calib_pair_pi is not None:
                li, ri, _ = self._pairs[self._calib_pair_pi]
                self._calib.maybe_capture(
                    stamps[li],
                    rect_all[self._calib_pair_pi * 2],       # left
                    rect_all[self._calib_pair_pi * 2 + 1])   # right

            self._work_queue.task_done()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--obstacle_debug', nargs='?', const='true', default='false',
        help='Publish constant obstacle scan to /mavros/obstacle/send every 100 ms.')
    
    parser.add_argument('--calib_capture', action='store_true')
    parser.add_argument('--calib_pair', default='1_0')
    parser.add_argument('--calib_bag_dir', default=None)
    parser.add_argument('--calib_ir_topic',
                        default='/camera/camera/infra1/image_rect_raw')
    # parser.add_argument('--calib_rgb_topic',
    #                     default='/camera/camera/color/image_raw')
    parser.add_argument('--calib_rgb_topic',
                        default='')   # empty = disabled
    parser.add_argument('--calib_max_dt_ms', type=float, default=20.0)

    args, unknown = parser.parse_known_args()
    
    calib_args = None
    if args.calib_capture:
        lp, rp = args.calib_pair.split('_')
        bag_dir = args.calib_bag_dir or os.path.expanduser(
            f'~/calib_capture_{datetime.now():%Y_%m_%d-%H_%M_%S}')
        calib_args = {
            'enabled': True,
            'pair': (int(lp), int(rp)),
            'bag_dir': bag_dir,
            'ir_topic': args.calib_ir_topic,
            'rgb_topic': args.calib_rgb_topic,
            'max_dt_ms': args.calib_max_dt_ms,
        }
    
    obstacle_debug = str(args.obstacle_debug).lower() in ('1', 'true', 'yes', 'on')
    rclpy.init(args=unknown)
    node = MultiStereoDisparityNode(obstacle_debug=obstacle_debug,
                                    calib_args=calib_args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_evt.set()
        node._gpu_thread.join(timeout=2.0)

        if getattr(node, '_calib', None) is not None:
            node._calib.shutdown()

        node.destroy_node()
        if rclpy.ok():                           # guard against double shutdown
            rclpy.shutdown()


if __name__ == '__main__':
    main()
