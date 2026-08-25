#!/usr/bin/env python3
"""Four-camera fisheye Depth Anything V2 TensorRT ROS 2 node.

Pipeline per frame (always a batch of 4, padding missing slots so the static
engine never sees a size mismatch; only fresh cameras are actually published):

  1. Decode 4 raw fisheye frames (CPU).
  2. GPU rectification: rectangular DoubleSphere inverse map per camera at the
     ENGINE's (H, W) -- square angular pixels; the larger dim spans HALF_ANGLE,
     the smaller dim scales proportionally. Output is device-resident.
  3. Sky segmentation (ncnn, CPU worker) runs in parallel on the rectified RGB.
  4. GPU prep: BGR->RGB, /255, ImageNet-normalize, NCHW.
  5. TRT inference -> (4, H, W) relative depth.
  6. Zero sky pixels, publish per fresh camera.
  7. Save a four-column rect/depth/sky/masked-depth JPEG per fresh frame.

Each camera uses the LEFT rectification map of the stereo pair in which it is
cam0 (cam 0 -> pair (0,3), cam 1 -> (1,0), cam 2 -> (2,1), cam 3 -> (3,2)), so
DA-V2 depth lands in the SAME rectified frame as the LightStereo left image.

Rectification and sky both degrade gracefully: missing deps / calib / model
just downgrade to plain resize / no-mask with a warning.

Sky refinement is enabled by default. Use --no-refine to keep EGE segmentation
but bypass confidence-weighted guided refinement; use --refine to enable it.
"""

import argparse
from pathlib import Path
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import torch
import torch.nn.functional as F
import yaml

from cv_bridge import CvBridge

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import ncnn
except ImportError:
    ncnn = None


# ===========================================================================
# Paths -- EDIT THESE for your machine.
# ===========================================================================
NODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = NODE_DIR.parent

DEFAULT_ENGINE = (
    PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits_b4_420x280.engine"
)

# Directory of the stereo node's rectify/ tree (rectify_utils.py,
# inverse_rect_ds.py, the calib yaml).
STEREO_RECTIFY_DIR = Path("/home/nvidia/ros_stereo/rectify")
CALIB_FILE = STEREO_RECTIFY_DIR / "stereo_calib_ds-camchain_from_json.yaml"

# ncnn sky model (EGE-UNet) -- same files the stereo node uses.
SKY_PARAM_FILE = STEREO_RECTIFY_DIR.parent / "EGE_165.ncnn.param"
SKY_BIN_FILE = STEREO_RECTIFY_DIR.parent / "EGE_165.ncnn.bin"

if STEREO_RECTIFY_DIR.is_dir():
    sys.path.insert(0, str(STEREO_RECTIFY_DIR))

# ===========================================================================
# Constants
# ===========================================================================
CAMERA_IDS = (0, 1, 2, 3)
BATCH_SIZE = len(CAMERA_IDS)

# Rectified / engine-input size. Must match the (H, W) baked into the engine.
INPUT_WIDTH = 420
INPUT_HEIGHT = 280

# Rectified-cone half-angle for the LARGER output dimension. The smaller dim
# gets HALF_ANGLE * (smaller / larger) so the angular pixel spacing is uniform
# (square pixels). Matches the stereo node's CUT_ANGLE = pi/4 convention.
CUT_ANGLE = np.pi / 4
HALF_ANGLE = np.pi / 2 - CUT_ANGLE     # ~0.785 rad; tan == 1.0

# Rectification toggle. Auto-disabled if deps / calib missing.
RECTIFY_ENABLE = True

# Sky segmentation.
SKY_ENABLE = True
SKY_INPUT_SIZE = 384                   # NCNN resize-input (model was trained square)
SKY_INPUT_BLOB = "in0"
SKY_OUTPUT_BLOB = "out5"              # final EGE_165 probability output
SKY_MEAN_VALS = [123.675, 116.28, 103.53]
SKY_NORM_VALS = [0.01712, 0.01751, 0.01743]
SKY_THRESHOLD = 0.5
SKY_USE_SIGMOID = False
SKY_CLASS_INDEX = 1
SKY_USE_VULKAN = False
SKY_NUM_THREADS = 2
SKY_OPENMP_BLOCKTIME = 0
SKY_CAMERAS = set(CAMERA_IDS)
SKY_FILL_VALUE = 0.0                   # value written into sky pixels of depth
SKY_PUBLISH_MASK = False               # publish /camera_{id}/sky_mask (mono8)

# Confidence-weighted guided refinement, ported from mask_refine.cpp.  The
# original factor 256 would collapse a 420x280 image to roughly 2x1, so use a
# working grid downsampled by 8 for this low-resolution deployment.
SKY_REFINE_ENABLE = True
SKY_REFINE_DOWNSAMPLE = 8
SKY_REFINE_LOW_THRESHOLD = 0.3
SKY_REFINE_HIGH_THRESHOLD = 0.5
SKY_REFINE_BIAS = 0.8
SKY_REFINE_EPSILON = 0.01
SKY_REFINE_REGULARIZATION = 0.01 ** 2
SKY_REFINE_BILATERAL_SIGMA_COLOR = 20.0
SKY_REFINE_BILATERAL_SIGMA_SPACE = 10.0

PUBLISH_RECT = False                   # publish /camera_{id}/image_rect (bgr8)

# Offline visual diagnostics.  Every fresh camera frame is saved; when disk
# writing falls behind, the inference worker blocks instead of dropping files.
SAVE_MONTAGE = True
MONTAGE_DIR = NODE_DIR / "img"
MONTAGE_QUEUE_SIZE = 8
MONTAGE_JPEG_QUALITY = 95

# ImageNet normalization (moved to CUDA in node __init__).
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406],
                          dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225],
                         dtype=torch.float32).view(1, 3, 1, 1)


def _ds_intrinsics(cam_cfg, s=1.0):
    """DoubleSphere intrinsics tuple (xi, alpha, fx, fy, cx, cy), scaled by s."""
    intr = cam_cfg["intrinsics"]
    return intr[0], intr[1], s * intr[2], s * intr[3], s * intr[4], s * intr[5]


# ---------------------------------------------------------------------------
# Rectification imports (guarded). We reuse the CUDA remap kernel and the plane
# geometry helpers from the stereo package.
# ---------------------------------------------------------------------------
_RECT_IMPORT_ERR = None
_REMAP_BILINEAR_KERNEL = None
_epipolar_fn = None
_plane_basis_fn = None
if RECTIFY_ENABLE and cp is not None:
    try:
        from inverse_rect_ds import _REMAP_BILINEAR_KERNEL as _KERNEL
        from rectify_utils import (
            epiploar_planes_from_extrinsics as _epi,
            create_plane_basis as _pbasis,
        )
        _REMAP_BILINEAR_KERNEL = _KERNEL
        _epipolar_fn = _epi
        _plane_basis_fn = _pbasis
    except Exception as exc:  # noqa: BLE001
        _RECT_IMPORT_ERR = exc


# ---------------------------------------------------------------------------
# Rectangular DoubleSphere inverse-map rectifier (mono, LEFT-cam only).
# ---------------------------------------------------------------------------
def _build_inverse_rect_map_rect_cuda(normal, u_axis, v_axis, ds_intr,
                                      n_h, n_w, b2n):
    """GPU rectangular inverse map with square angular pixels.

    The larger of (n_h, n_w) spans [-tan(b2n), tan(b2n)]; the smaller dim
    scales proportionally so the tangent-space pixel step is identical on both
    axes. Output cells that fall outside the DoubleSphere projectable domain
    are marked NaN so the remap kernel blacks them out.
    """
    xi, alpha, fx, fy, cx, cy = ds_intr
    normal = cp.asarray(normal, cp.float32)
    u_axis = cp.asarray(u_axis, cp.float32)
    v_axis = cp.asarray(v_axis, cp.float32)

    if n_w >= n_h:
        hw_x = float(np.tan(b2n))
        hw_y = hw_x * (n_h / n_w)
    else:
        hw_y = float(np.tan(b2n))
        hw_x = hw_y * (n_w / n_h)

    grid_x = cp.linspace(-hw_x,  hw_x, n_w, dtype=cp.float32)   # column axis
    grid_y = cp.linspace( hw_y, -hw_y, n_h, dtype=cp.float32)   # row axis, top -> +v
    uu, vv = cp.meshgrid(grid_x, grid_y)

    X = (normal[None, None, :]
         + uu[..., None] * u_axis[None, None, :]
         + vv[..., None] * v_axis[None, None, :]).reshape(-1, 3)

    x = X[:, 0]; y = X[:, 1]; z = X[:, 2]
    d1 = cp.sqrt(x * x + y * y + z * z)
    k = xi * d1 + z
    d2 = cp.sqrt(x * x + y * y + k * k)
    denom = alpha * d2 + (1.0 - alpha) * k

    if alpha <= 0.5:
        w1 = alpha / (1.0 - alpha)
    else:
        w1 = (1.0 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    valid = (z > -w2 * d1) & (denom > 1e-9)

    nan = cp.float32(cp.nan)
    map_x = cp.where(valid, fx * x / denom + cx, nan).astype(cp.float32)
    map_y = cp.where(valid, fy * y / denom + cy, nan).astype(cp.float32)
    return cp.ascontiguousarray(map_x), cp.ascontiguousarray(map_y)


class FisheyeMonoRectifierRectCUDA:
    """Single-camera GPU rectifier (LEFT map of a stereo pair) at (n_h, n_w).

    Build-once / remap-per-frame flow -- calib -> device-resident inverse maps;
    per frame: one H2D of the raw fisheye BGR, one remap kernel, output stays
    on GPU (or downloads to host if download=True).
    """

    def __init__(self, calib_pair, n_h, n_w, half_angle,
                 ds_intrinsics_fn, epipolar_fn, plane_basis_fn, logger=None):
        if cp is None or _REMAP_BILINEAR_KERNEL is None:
            raise RuntimeError("FisheyeMonoRectifierRectCUDA requires CuPy + kernel")

        def log(msg):
            if logger is not None:
                logger.info(msg)

        cam0_cfg = calib_pair["cam0"]
        cam1_cfg = calib_pair["cam1"]
        cW, cH = cam0_cfg["resolution"]
        sW, sH = calib_pair.get("stream_resolution", (cW, cH))
        scale = sH / cH

        ds0 = ds_intrinsics_fn(cam0_cfg, scale)

        R_mat = np.array(cam1_cfg["R"], dtype=np.float64)
        t_vec = np.array(cam1_cfg["t"], dtype=np.float64)
        R = R_mat.T
        t = R_mat @ t_vec

        n0, _n1 = epipolar_fn(R, t)
        u0, v0 = plane_basis_fn(n0, t)

        log(f"    Building GPU rectangular map ({n_h}x{n_w}, DoubleSphere)...")
        self.map_x, self.map_y = _build_inverse_rect_map_rect_cuda(
            n0, u0, v0, ds0, n_h, n_w, half_angle)

        self.n_h = int(n_h)
        self.n_w = int(n_w)
        self.n_out = self.n_h * self.n_w
        self.H_src = int(sH)
        self.W_src = int(sW)

        self._src_buf = cp.empty(self.H_src * self.W_src * 3, dtype=cp.uint8)
        self._out_buf = cp.empty(self.n_out * 3, dtype=cp.uint8)
        self._threads = 256
        self._blocks = (self.n_out + self._threads - 1) // self._threads

        n_valid = int(cp.sum(~cp.isnan(self.map_x)))
        log(f"    valid output cells: {n_valid}/{self.n_out}")

    def rectify(self, src_bgr, download=False):
        """(H_src, W_src, 3) uint8 BGR -> (n_h, n_w, 3) uint8 BGR.

        With download=False returns a device-resident CuPy view of the same
        internal buffer; the caller must consume it before the next call.
        """
        if isinstance(src_bgr, cp.ndarray):
            self._src_buf[...] = src_bgr.ravel()
        else:
            self._src_buf.set(np.ascontiguousarray(src_bgr).ravel())

        _REMAP_BILINEAR_KERNEL(
            (self._blocks,), (self._threads,),
            (self._src_buf, self.map_x, self.map_y,
             np.int32(self.H_src), np.int32(self.W_src), np.int32(self.n_out),
             self._out_buf),
        )
        if download:
            return self._out_buf.get().reshape(self.n_h, self.n_w, 3)
        return self._out_buf.reshape(self.n_h, self.n_w, 3)


# ---------------------------------------------------------------------------
def _image_msg_to_bgr(message: Image) -> np.ndarray:
    conversions = {
        "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "bgra8": (4, cv2.COLOR_BGRA2BGR),
        "rgba8": (4, cv2.COLOR_RGBA2BGR),
        "mono8": (1, cv2.COLOR_GRAY2BGR),
    }
    encoding = message.encoding.lower()
    if encoding not in conversions:
        raise ValueError(
            f"Unsupported encoding {message.encoding!r}; expected one of "
            f"{sorted(conversions)}")
    channels, conversion = conversions[encoding]
    row_bytes = int(message.width) * channels
    if int(message.step) < row_bytes:
        raise ValueError(
            f"Invalid Image step {message.step} for {message.width}x{channels}")
    required_bytes = int(message.step) * int(message.height)
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    if buffer.size < required_bytes:
        raise ValueError(
            f"Image buffer has {buffer.size} bytes; expected {required_bytes}")
    rows = buffer[:required_bytes].reshape(message.height, message.step)
    pixels = rows[:, :row_bytes]
    if channels == 1:
        image = pixels.reshape(message.height, message.width)
    else:
        image = pixels.reshape(message.height, message.width, channels)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return np.ascontiguousarray(image)


# ---------------------------------------------------------------------------
# Sky segmentation (ncnn) -- ported from the stereo node.
# ---------------------------------------------------------------------------
class NCNNSkySegmenter:
    """Run EGE-UNet and return an (H, W) float32 sky probability map.

    The model was trained square; input images are resized to SKY_INPUT_SIZE
    internally, and the output probability is resized back to the input HxW.
    """

    def __init__(self, param_path, bin_path, input_size, in_blob, out_blob,
                 mean_vals, norm_vals, threshold, use_sigmoid, class_index,
                 use_vulkan=False, num_threads=0, openmp_blocktime=0):
        if ncnn is None:
            raise RuntimeError("ncnn is not importable (pip install ncnn)")
        param_path, bin_path = str(param_path), str(bin_path)
        if not (Path(param_path).is_file() and Path(bin_path).is_file()):
            raise FileNotFoundError(f"sky model missing: {param_path} / {bin_path}")
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = bool(use_vulkan)
        if num_threads and num_threads > 0:
            self.net.opt.num_threads = int(num_threads)
        try:
            self.net.opt.openmp_blocktime = int(openmp_blocktime)
        except AttributeError:
            pass
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        self.input_size = int(input_size)
        self.in_blob = in_blob
        self.out_blob = out_blob
        self.mean_vals = list(mean_vals)
        self.norm_vals = list(norm_vals)
        self.threshold = float(threshold)
        self.use_sigmoid = bool(use_sigmoid)
        self.class_index = int(class_index)
        self.num_threads = int(num_threads) if num_threads else 0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mat_in = ncnn.Mat.from_pixels_resize(
            rgb, ncnn.Mat.PixelType.PIXEL_RGB, W, H,
            self.input_size, self.input_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        ex = self.net.create_extractor()
        if self.num_threads > 0:
            try:
                ex.set_num_threads(self.num_threads)
            except AttributeError:
                pass
        ex.input(self.in_blob, mat_in)
        ret, mat_out = ex.extract(self.out_blob)
        if ret != 0:
            raise RuntimeError(
                f"ncnn extract failed (ret={ret}) on blob {self.out_blob!r}")
        out = np.array(mat_out)
        if out.ndim == 2:
            prob = out
        elif out.shape[0] == 1:
            prob = out[0]
        else:
            e = np.exp(out - out.max(axis=0, keepdims=True))
            prob = (e / e.sum(axis=0, keepdims=True))[self.class_index]
        if self.use_sigmoid and out.ndim <= 3 and (
                out.ndim == 2 or out.shape[0] == 1):
            prob = 1.0 / (1.0 + np.exp(-prob))
        prob = np.asarray(prob, dtype=np.float32).squeeze()
        if prob.ndim != 2:
            raise ValueError(f"Unexpected sky output shape {out.shape}")
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0).astype(np.float32)


def _schlick_bias(values: np.ndarray, bias_value: float) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    denominator = ((1.0 / bias_value) - 2.0) * (1.0 - values) + 1.0
    return values / np.maximum(denominator, 1e-6)


def _probability_to_confidence(probability: np.ndarray) -> np.ndarray:
    """Port mask_refine.cpp probability_to_confidence(), vectorized."""
    probability = np.clip(probability, 0.0, 1.0).astype(np.float32)
    confidence = np.full(
        probability.shape, SKY_REFINE_EPSILON, dtype=np.float32)

    low = probability < SKY_REFINE_LOW_THRESHOLD
    if np.any(low):
        low_values = (
            SKY_REFINE_LOW_THRESHOLD - probability[low]
        ) / SKY_REFINE_LOW_THRESHOLD
        confidence[low] = np.maximum(
            SKY_REFINE_EPSILON,
            _schlick_bias(low_values, SKY_REFINE_BIAS),
        )

    high = probability > SKY_REFINE_HIGH_THRESHOLD
    if np.any(high):
        high_values = (
            probability[high] - SKY_REFINE_HIGH_THRESHOLD
        ) / (1.0 - SKY_REFINE_HIGH_THRESHOLD)
        confidence[high] = np.maximum(
            SKY_REFINE_EPSILON,
            _schlick_bias(high_values, SKY_REFINE_BIAS),
        )
    return confidence


def _self_resize(array: np.ndarray, target_hw) -> np.ndarray:
    """Antialiased downsample followed by linear resize, as in the C++ code."""
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    output = np.asarray(array, dtype=np.float32)
    if output.ndim == 3 and output.shape[2] > 4:
        return np.stack(
            [_self_resize(output[..., channel], target_hw)
             for channel in range(output.shape[2])],
            axis=-1,
        )
    kernel = np.asarray([1 / 8, 3 / 8, 3 / 8, 1 / 8], dtype=np.float32)
    while output.shape[1] >= 2 * target_w and output.shape[0] >= 2 * target_h:
        output = cv2.sepFilter2D(
            output, -1, kernel, kernel,
            anchor=(1, 1), borderType=cv2.BORDER_REPLICATE)
        next_w = max(1, int(round(output.shape[1] / 2.0)))
        next_h = max(1, int(round(output.shape[0] / 2.0)))
        output = cv2.pyrDown(output, dstsize=(next_w, next_h))
    return cv2.resize(
        output, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _weighted_downsample(
        array: np.ndarray, confidence: np.ndarray, target_hw) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        weighted = array * confidence
    else:
        weighted = array * confidence[..., None]
    numerator = _self_resize(weighted, target_hw)
    denominator = _self_resize(confidence, target_hw)
    if numerator.ndim == 2:
        return numerator / np.maximum(denominator, 1e-6)
    return numerator / np.maximum(denominator[..., None], 1e-6)


def _smooth_upsample(array: np.ndarray, target_hw) -> np.ndarray:
    """Progressive linear upsampling equivalent to smooth_upsample()."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    output = np.asarray(array, dtype=np.float32)
    ratio = max(target_w / output.shape[1], target_h / output.shape[0])
    steps = max(1, int(round(0.5 * np.log2(max(ratio, 1.0)))))
    for step in range(1, steps + 1):
        width = max(1, int(round(target_w * step / steps)))
        height = max(1, int(round(target_h * step / steps)))
        output = cv2.resize(
            output, (width, height), interpolation=cv2.INTER_LINEAR)
    return output


def _refine_sky_probability(
        rgb: np.ndarray, coarse_probability: np.ndarray,
        enabled: bool = SKY_REFINE_ENABLE) -> np.ndarray:
    """Confidence-weighted local RGB affine refinement from mask_refine.cpp."""
    probability = np.clip(
        coarse_probability, 0.0, 1.0).astype(np.float32)
    if not enabled:
        return probability

    reference = np.asarray(rgb, dtype=np.float32) / 255.0
    height, width = probability.shape
    target_h = max(2, int(round(height / SKY_REFINE_DOWNSAMPLE)))
    target_w = max(2, int(round(width / SKY_REFINE_DOWNSAMPLE)))
    target_hw = (target_h, target_w)
    confidence = _probability_to_confidence(probability)

    reference_small = _weighted_downsample(
        reference, confidence, target_hw)
    source_small = _weighted_downsample(
        probability, confidence, target_hw)

    r, g, b = cv2.split(reference)
    moments = np.stack((r * r, r * g, r * b, g * g, g * b, b * b), axis=-1)
    moments_small = _weighted_downsample(moments, confidence, target_hw)
    rs, gs, bs = cv2.split(reference_small)

    covariance = np.empty((target_h, target_w, 3, 3), dtype=np.float32)
    covariance[..., 0, 0] = moments_small[..., 0] - rs * rs
    covariance[..., 0, 1] = moments_small[..., 1] - rs * gs
    covariance[..., 0, 2] = moments_small[..., 2] - rs * bs
    covariance[..., 1, 0] = covariance[..., 0, 1]
    covariance[..., 1, 1] = moments_small[..., 3] - gs * gs
    covariance[..., 1, 2] = moments_small[..., 4] - gs * bs
    covariance[..., 2, 0] = covariance[..., 0, 2]
    covariance[..., 2, 1] = covariance[..., 1, 2]
    covariance[..., 2, 2] = moments_small[..., 5] - bs * bs
    diagonal = np.arange(3)
    covariance[..., diagonal, diagonal] += SKY_REFINE_REGULARIZATION

    reference_source = reference * probability[..., None]
    reference_source_small = _weighted_downsample(
        reference_source, confidence, target_hw)
    cross_covariance = (
        reference_source_small
        - reference_small * source_small[..., None]
    )

    matrices = covariance.reshape(-1, 3, 3)
    vectors = cross_covariance.reshape(-1, 3, 1)
    try:
        affine = np.linalg.solve(matrices, vectors).reshape(target_h, target_w, 3)
    except np.linalg.LinAlgError:
        affine = np.matmul(
            np.linalg.pinv(matrices), vectors).reshape(target_h, target_w, 3)
    residual = source_small - np.sum(affine * reference_small, axis=2)

    affine_full = _smooth_upsample(affine, (height, width))
    residual_full = _smooth_upsample(residual, (height, width))
    refined = np.clip(
        np.sum(reference * affine_full, axis=2) + residual_full,
        0.0, 1.0).astype(np.float32)

    refined_255 = cv2.bilateralFilter(
        refined * 255.0,
        d=0,
        sigmaColor=SKY_REFINE_BILATERAL_SIGMA_COLOR,
        sigmaSpace=SKY_REFINE_BILATERAL_SIGMA_SPACE,
    )
    return np.clip(refined_255 / 255.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# TensorRT engine
# ---------------------------------------------------------------------------
class TensorRTEngine:
    """TensorRT 10 engine runner using PyTorch CUDA buffers (fully static)."""

    _DTYPE_MAP = {
        np.dtype("float32"): torch.float32,
        np.dtype("float16"): torch.float16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
    }

    def __init__(self, engine_path: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires PyTorch CUDA support")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT Python bindings are not installed") from exc

        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        with engine_path.open("rb") as stream:
            self._engine = self._runtime.deserialize_cuda_engine(stream.read())
        if self._engine is None:
            raise RuntimeError(f"Unable to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Unable to create TensorRT execution context")

        self._input_names, self._output_names = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
        if len(self._input_names) != 1 or len(self._output_names) != 1:
            raise RuntimeError(
                "Expected one TensorRT input and one output; got "
                f"inputs={self._input_names}, outputs={self._output_names}")
        self.input_name = self._input_names[0]
        self.output_name = self._output_names[0]
        self._buffers = {}
        self._stream = torch.cuda.Stream()
        self.input_shape = tuple(self._engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self._engine.get_tensor_shape(self.output_name))

    def _ensure_output_buffer(self):
        if self.output_name in self._buffers:
            return
        shape = self.output_shape
        if any(d < 0 for d in shape):
            raise RuntimeError(
                f"infer_gpu requires a fully-static engine; output shape {shape}")
        np_dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(self.output_name)))
        torch_dtype = self._DTYPE_MAP[np_dtype]
        self._buffers[self.output_name] = torch.empty(
            shape, dtype=torch_dtype, device="cuda").contiguous()
        self._context.set_tensor_address(
            self.output_name, self._buffers[self.output_name].data_ptr())

    def infer_gpu(self, gpu_tensor: torch.Tensor) -> np.ndarray:
        self._ensure_output_buffer()
        gpu_tensor = gpu_tensor.contiguous()
        self._context.set_tensor_address(self.input_name, gpu_tensor.data_ptr())

        # Preprocessing and the non-blocking copy into gpu_tensor run on
        # PyTorch's current stream, while TensorRT uses its own stream.  Make
        # the TensorRT consumer wait for that producer so it cannot infer from
        # a previous or only partially updated batch.
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        self._stream.synchronize()
        out = self._buffers[self.output_name]
        if out.ndim == 4 and out.shape[1] == 1:
            out = out[:, 0]
        return out.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class DepthAnythingV2TensorRTNode(Node):
    def __init__(self, engine_path: Path, refine_enabled: bool):
        super().__init__("depth_anything_v2_fisheye_trt")
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        self._engine = TensorRTEngine(engine_path)
        _, _, eng_h, eng_w = self._engine.input_shape
        self._eng_h, self._eng_w = int(eng_h), int(eng_w)
        self.get_logger().info(
            f"Engine I/O: input={self._engine.input_name}{self._engine.input_shape}, "
            f"output={self._engine.output_name}{self._engine.output_shape}")
        if (self._eng_h, self._eng_w) != (INPUT_HEIGHT, INPUT_WIDTH):
            self.get_logger().warn(
                f"Engine input ({self._eng_h}x{self._eng_w}) != "
                f"({INPUT_HEIGHT}x{INPUT_WIDTH}); rectified images will be "
                f"resized before inference and depth resized back after.")

        # Rectifier + publish grid: match the engine exactly.
        self.rect_h = self._eng_h
        self.rect_w = self._eng_w

        self._keys = list(CAMERA_IDS)
        self._bridge = CvBridge()
        self._sky_refine_enabled = bool(refine_enabled)

        # ---- Preprocessing buffers ----
        # Persistent rectified BGR batch (BATCH, rect_h, rect_w, 3) uint8 on CUDA.
        self._rect_gpu = torch.empty(
            (BATCH_SIZE, self.rect_h, self.rect_w, 3),
            dtype=torch.uint8, device="cuda")
        # Persistent engine input (BATCH, 3, eng_h, eng_w) float32 on CUDA.
        self._input_gpu = torch.empty(
            (BATCH_SIZE, 3, self._eng_h, self._eng_w),
            dtype=torch.float32, device="cuda").contiguous()
        self._norm_mean = _NORM_MEAN.cuda()
        self._norm_std = _NORM_STD.cuda()

        # ---- Rectifiers (one LEFT map per camera) ----
        self._rectifiers = {}       # cam_id -> FisheyeMonoRectifierRectCUDA
        self._rect_src_hw = {}
        if RECTIFY_ENABLE:
            self._setup_rectifiers()

        # ---- Sky segmenter + worker ----
        self._sky_seg = None
        self._sky_in_q = None
        self._sky_out_q = None
        self._sky_thread = None
        self._stop_worker = False
        if SKY_ENABLE:
            self._setup_sky()

        # ---- QoS + slots ----
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self._depth_publishers = {}
        self._rect_publishers = {}
        self._sky_publishers = {}
        self._image_subscriptions = []
        self._latest_frames = {k: None for k in self._keys}
        self._fresh_frames = set()
        self._slot_condition = threading.Condition()
        self._last_statistics_log = 0.0

        # ---- Offline montage writer ----
        self._montage_queue = None
        self._montage_thread = None
        if SAVE_MONTAGE:
            for camera_id in self._keys:
                (MONTAGE_DIR / f"camera_{camera_id}").mkdir(
                    parents=True, exist_ok=True)
            self._montage_queue = queue.Queue(maxsize=MONTAGE_QUEUE_SIZE)
            self._montage_thread = threading.Thread(
                target=self._montage_writer,
                name="montage_writer",
                daemon=True,
            )
            self._montage_thread.start()

        for camera_id in self._keys:
            in_topic = f"/synced/camera_{camera_id}/image_raw" #temp
            out_topic = f"/camera_{camera_id}/relative_depth"
            self._depth_publishers[camera_id] = self.create_publisher(Image, out_topic, 1)
            if PUBLISH_RECT:
                self._rect_publishers[camera_id] = self.create_publisher(
                    Image, f"/camera_{camera_id}/image_rect", 1)
            if SKY_PUBLISH_MASK and self._sky_seg is not None and camera_id in SKY_CAMERAS:
                self._sky_publishers[camera_id] = self.create_publisher(
                    Image, f"/camera_{camera_id}/sky_mask", 1)
            sub = self.create_subscription(
                Image, in_topic,
                lambda message, slot_key=camera_id: self._image_callback(slot_key, message),
                qos)
            self._image_subscriptions.append(sub)
            self.get_logger().info(f"  {in_topic} -> {out_topic}")

        self._worker = threading.Thread(
            target=self._gpu_worker, name="da_v2_gpu_worker", daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"Ready: {BATCH_SIZE} cams; rect="
            f"{'on' if self._rectifiers else 'off'}, "
            f"sky={'on' if self._sky_seg else 'off'}, "
            f"refine={'on' if self._sky_refine_enabled else 'off'}; "
            f"rect {self.rect_h}x{self.rect_w} == engine "
            f"{self._eng_h}x{self._eng_w}; "
            f"montage={'on: ' + str(MONTAGE_DIR) if SAVE_MONTAGE else 'off'}")

    # ------------------------------------------------------------------
    @staticmethod
    def _depth_colormap(depth, valid_mask):
        valid = valid_mask & np.isfinite(depth)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            lo = float(depth[valid].min())
            hi = float(depth[valid].max())
            if hi > lo:
                normalized = np.nan_to_num(
                    (depth - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
                gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    def _queue_montage(
            self, cam, source_message, rect_bgr, raw_depth,
            sky_mask, refined_alpha):
        if self._montage_queue is None:
            return
        if sky_mask is None or sky_mask.shape != raw_depth.shape:
            sky_mask = np.zeros(raw_depth.shape, dtype=bool)
        if refined_alpha is None or refined_alpha.shape != raw_depth.shape:
            refined_alpha = np.zeros(raw_depth.shape, dtype=np.float32)
        valid = ~sky_mask
        raw_color = self._depth_colormap(raw_depth, valid)
        masked_color = raw_color.copy()
        masked_color[sky_mask] = 0
        alpha_u8 = np.clip(refined_alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR)
        montage = np.hstack((rect_bgr, raw_color, alpha_bgr, masked_color))
        stamp = source_message.header.stamp
        path = MONTAGE_DIR / f"camera_{cam}" / f"{stamp.sec}_{stamp.nanosec:09d}.jpg"
        self._montage_queue.put((path, np.ascontiguousarray(montage)))

    def _montage_writer(self):
        while True:
            item = self._montage_queue.get()
            try:
                if item is None:
                    return
                path, montage = item
                ok = cv2.imwrite(
                    str(path), montage,
                    [cv2.IMWRITE_JPEG_QUALITY, MONTAGE_JPEG_QUALITY],
                )
                if not ok:
                    self.get_logger().error(f"Unable to save montage: {path}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Montage write failed: {exc}")
            finally:
                self._montage_queue.task_done()

    # ------------------------------------------------------------------
    def _setup_rectifiers(self):
        if cp is None or _REMAP_BILINEAR_KERNEL is None:
            self.get_logger().warn(
                f"Rectification disabled (deps missing): {_RECT_IMPORT_ERR}")
            return
        if not CALIB_FILE.is_file():
            self.get_logger().warn(
                f"Rectification disabled: calib not found at {CALIB_FILE}")
            return
        try:
            with CALIB_FILE.open() as f:
                all_calib = yaml.safe_load(f)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Rectification disabled: calib load failed ({exc})")
            return

        pair_re = re.compile(r"^cam_pair_(\d+)_(\d+)$")
        for key in sorted(all_calib.keys()):
            m = pair_re.match(key)
            if not m:
                continue
            left_idx = int(m.group(1))
            if left_idx not in self._keys:
                continue
            try:
                rect = FisheyeMonoRectifierRectCUDA(
                    all_calib[key], self.rect_h, self.rect_w, HALF_ANGLE,
                    _ds_intrinsics, _epipolar_fn, _plane_basis_fn,
                    logger=self.get_logger())
                self._rectifiers[left_idx] = rect
                self._rect_src_hw[left_idx] = (rect.H_src, rect.W_src)
                self.get_logger().info(
                    f"  Rectifier cam {left_idx}: {key} left-map "
                    f"(src {rect.H_src}x{rect.W_src} -> "
                    f"{self.rect_h}x{self.rect_w})")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"  Rectifier cam {left_idx} failed ({exc}); plain resize fallback")

        missing = [c for c in self._keys if c not in self._rectifiers]
        if missing:
            self.get_logger().warn(
                f"No LEFT rectifier for cameras {missing}; those use plain resize")

    # ------------------------------------------------------------------
    def _setup_sky(self):
        if ncnn is None:
            self.get_logger().warn("Sky disabled: ncnn not importable")
            return
        try:
            self._sky_seg = NCNNSkySegmenter(
                SKY_PARAM_FILE, SKY_BIN_FILE, SKY_INPUT_SIZE,
                SKY_INPUT_BLOB, SKY_OUTPUT_BLOB, SKY_MEAN_VALS, SKY_NORM_VALS,
                SKY_THRESHOLD, SKY_USE_SIGMOID, SKY_CLASS_INDEX, SKY_USE_VULKAN,
                num_threads=SKY_NUM_THREADS, openmp_blocktime=SKY_OPENMP_BLOCKTIME)
            self._sky_in_q = queue.Queue(maxsize=1)
            self._sky_out_q = queue.Queue(maxsize=1)
            self._sky_thread = threading.Thread(
                target=self._sky_worker, name="sky_worker", daemon=True)
            self._sky_thread.start()
            self.get_logger().info(
                f"Sky segmenter loaded; cameras={sorted(SKY_CAMERAS)}; "
                f"threads={SKY_NUM_THREADS} blocktime={SKY_OPENMP_BLOCKTIME}ms")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"Sky init failed ({exc}); continuing without masking")
            self._sky_seg = None

    # ------------------------------------------------------------------
    def _sky_worker(self):
        """Run coarse EGE and guided refinement while the GPU pipeline runs."""
        while True:
            item = self._sky_in_q.get()
            if item is None:
                break
            rgb_by_cam, seq = item
            results = {}
            for cam, rgb in rgb_by_cam.items():
                try:
                    started = time.perf_counter()
                    coarse = self._sky_seg(rgb)
                    segmented = time.perf_counter()
                    refined = _refine_sky_probability(
                        rgb, coarse, enabled=self._sky_refine_enabled)
                    finished = time.perf_counter()
                    binary = refined > SKY_THRESHOLD
                    results[cam] = {
                        "mask": binary,
                        "alpha": refined,
                        "coarse_min": float(coarse.min()),
                        "coarse_max": float(coarse.max()),
                        "refined_min": float(refined.min()),
                        "refined_max": float(refined.max()),
                        "sky_pixels": int(np.count_nonzero(binary)),
                        "ege_ms": (segmented - started) * 1000.0,
                        "refine_ms": (finished - segmented) * 1000.0,
                    }
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"sky seg failed cam {cam}: {exc}",
                                           throttle_duration_sec=2.0)
                    zeros = np.zeros(rgb.shape[:2], dtype=np.float32)
                    results[cam] = {
                        "mask": zeros.astype(bool),
                        "alpha": zeros,
                        "coarse_min": 0.0,
                        "coarse_max": 0.0,
                        "refined_min": 0.0,
                        "refined_max": 0.0,
                        "sky_pixels": 0,
                        "ege_ms": 0.0,
                        "refine_ms": 0.0,
                    }
            self._sky_out_q.put((results, seq))

    def _submit_sky_async(self, rgb_by_cam, seq):
        if self._sky_seg is None or not rgb_by_cam:
            return False
        # Offline capture: wait for queue space rather than dropping a frame.
        self._sky_in_q.put((rgb_by_cam, seq))
        return True

    def _join_sky(self, seq):
        if self._sky_seg is None:
            return {}
        # Offline capture: refinement is required, so wait for this exact batch.
        results, mask_seq = self._sky_out_q.get()
        if mask_seq != seq:
            raise RuntimeError(
                f"sky mask seq mismatch got={mask_seq} want={seq}")
        return results

    # ------------------------------------------------------------------
    def _image_callback(self, key, message: Image):
        with self._slot_condition:
            self._latest_frames[key] = message
            self._fresh_frames.add(key)
            self._slot_condition.notify()

    def _next_batch(self):
        with self._slot_condition:
            while not self._stop_worker and not self._fresh_frames:
                self._slot_condition.wait()
            if self._stop_worker:
                return None, None, None
            fresh = set(self._fresh_frames)
            self._fresh_frames.clear()
            fallback = next((self._latest_frames[k] for k in self._keys
                             if self._latest_frames[k] is not None), None)
            if fallback is None:
                return None, None, None
            messages, real_mask = [], []
            for k in self._keys:
                m = self._latest_frames[k]
                if m is None:
                    messages.append(fallback)
                    real_mask.append(False)
                else:
                    messages.append(m)
                    real_mask.append(k in fresh)
            return list(self._keys), messages, real_mask

    # ------------------------------------------------------------------
    def _rectify_into_batch(self, keys, messages, real_mask):
        """Decode + GPU-rectify (or plain resize) into self._rect_gpu.

        Returns {cam_id: rgb_rect} for fresh cameras (fed to the sky worker).
        """
        sky_rgb = {}
        for i, (cam, msg) in enumerate(zip(keys, messages)):
            bgr = _image_msg_to_bgr(msg)
            rect = self._rectifiers.get(cam)
            if rect is not None:
                sh, sw = self._rect_src_hw[cam]
                if bgr.shape[:2] != (sh, sw):
                    bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
                rect_view = rect.rectify(bgr, download=False)         # CuPy
                t_view = torch.from_dlpack(rect_view.toDlpack())      # (H,W,3) uint8 CUDA
                self._rect_gpu[i].copy_(t_view, non_blocking=True)
            else:
                if bgr.shape[:2] != (self.rect_h, self.rect_w):
                    bgr = cv2.resize(bgr, (self.rect_w, self.rect_h),
                                     interpolation=cv2.INTER_AREA)
                self._rect_gpu[i].copy_(
                    torch.from_numpy(np.ascontiguousarray(bgr)), non_blocking=True)

            if real_mask[i] and self._sky_seg is not None and cam in SKY_CAMERAS:
                # ncnn is CPU, so this D2H is unavoidable. Kept small (rect size).
                rect_bgr_cpu = self._rect_gpu[i].cpu().numpy()
                sky_rgb[cam] = cv2.cvtColor(rect_bgr_cpu, cv2.COLOR_BGR2RGB)
        return sky_rgb

    def _prep_from_rect(self) -> torch.Tensor:
        """self._rect_gpu (B,H,W,3 uint8 BGR) -> self._input_gpu (B,3,H,W) float32."""
        rgb = self._rect_gpu[..., [2, 1, 0]].float().mul_(1.0 / 255.0)  # (B,H,W,3)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()                     # (B,3,H,W)
        rgb.sub_(self._norm_mean).div_(self._norm_std)
        if (rgb.shape[2], rgb.shape[3]) != (self._eng_h, self._eng_w):
            rgb = F.interpolate(rgb, size=(self._eng_h, self._eng_w),
                                mode="bilinear", align_corners=False)
        self._input_gpu.copy_(rgb, non_blocking=True)
        return self._input_gpu

    # ------------------------------------------------------------------
    def _gpu_worker(self):
        seq = 0
        while True:
            keys, messages, real_mask = self._next_batch()
            if not messages:
                return
            seq += 1
            self._infer_and_publish_batch(keys, messages, real_mask, seq)

    def _infer_and_publish_batch(self, keys, messages, real_mask, seq):
        try:
            t0 = time.perf_counter()

            sky_rgb = self._rectify_into_batch(keys, messages, real_mask)
            submitted = self._submit_sky_async(sky_rgb, seq)

            gpu_input = self._prep_from_rect()
            t_prep = time.perf_counter()

            depth_batch = self._engine.infer_gpu(gpu_input)   # (B, eng_h, eng_w)
            t_infer = time.perf_counter()

            # Resize depth to rect grid if engine differs.
            if depth_batch.shape[1:] != (self.rect_h, self.rect_w):
                depth_batch = np.stack([
                    cv2.resize(depth_batch[i], (self.rect_w, self.rect_h),
                               interpolation=cv2.INTER_LINEAR)
                    for i in range(depth_batch.shape[0])])

            sky_results = self._join_sky(seq) if submitted else {}
            t_sky = time.perf_counter()

            n_pub = 0
            for i, cam in enumerate(keys):
                if not real_mask[i]:
                    continue
                raw_depth = np.ascontiguousarray(
                    depth_batch[i], dtype=np.float32).copy()
                depth = raw_depth.copy()
                sky_result = sky_results.get(cam)
                mask = sky_result["mask"] if sky_result is not None else None
                refined_alpha = (
                    sky_result["alpha"] if sky_result is not None else None)
                if mask is not None and mask.shape == depth.shape:
                    depth[mask] = SKY_FILL_VALUE
                self._publish_image(depth, "32FC1", messages[i],
                                    self._depth_publishers[cam])
                rect_bgr = None
                if cam in self._rect_publishers:
                    rect_bgr = self._rect_gpu[i].cpu().numpy()
                    self._publish_image(rect_bgr, "bgr8", messages[i],
                                        self._rect_publishers[cam])
                if cam in self._sky_publishers and mask is not None:
                    self._publish_image((mask.astype(np.uint8) * 255), "mono8",
                                        messages[i], self._sky_publishers[cam])
                if self._montage_queue is not None:
                    if rect_bgr is None:
                        rect_bgr = self._rect_gpu[i].cpu().numpy()
                    self._queue_montage(
                        cam, messages[i], rect_bgr, raw_depth,
                        mask, refined_alpha)
                n_pub += 1
            t_pub = time.perf_counter()

            now = time.monotonic()
            if now - self._last_statistics_log >= 2.0:
                real_keys = [k for k, r in zip(keys, real_mask) if r]
                total_ms = (t_pub - t0) * 1000.0
                sky_values = list(sky_results.values())
                if sky_values:
                    ege_ms = sum(v["ege_ms"] for v in sky_values)
                    refine_ms = sum(v["refine_ms"] for v in sky_values)
                    sky_pixels = sum(v["sky_pixels"] for v in sky_values)
                    sky_total = sum(v["mask"].size for v in sky_values)
                    coarse_min = min(v["coarse_min"] for v in sky_values)
                    coarse_max = max(v["coarse_max"] for v in sky_values)
                    refined_min = min(v["refined_min"] for v in sky_values)
                    refined_max = max(v["refined_max"] for v in sky_values)
                    sky_detail = (
                        f" | ege={ege_ms:.2f} ms refine={refine_ms:.2f} ms "
                        f"coarse=[{coarse_min:.3f},{coarse_max:.3f}] "
                        f"refined=[{refined_min:.3f},{refined_max:.3f}] "
                        f"sky={sky_pixels}/{sky_total}")
                else:
                    sky_detail = " | sky=off"
                self.get_logger().info(
                    f"batch={BATCH_SIZE} real={real_keys} ({n_pub} pub) | "
                    f"prep={(t_prep - t0) * 1000.0:.2f} ms, "
                    f"infer={(t_infer - t_prep) * 1000.0:.2f} ms, "
                    f"sky={(t_sky - t_infer) * 1000.0:.2f} ms, "
                    f"pub={(t_pub - t_sky) * 1000.0:.2f} ms, "
                    f"total={total_ms:.2f} ms ({1000.0 / total_ms:.1f} Hz/batch)"
                    f"{sky_detail}")
                self._last_statistics_log = now
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Depth Anything batch failed for {keys}: {exc}")

    # ------------------------------------------------------------------
    def destroy_node(self):
        with self._slot_condition:
            self._stop_worker = True
            self._slot_condition.notify_all()
        if hasattr(self, "_worker"):
            # Offline capture requires every queued frame to be written, so
            # wait for the inference worker rather than abandoning its last
            # montage after an arbitrary timeout.
            self._worker.join()
        if self._sky_thread is not None:
            self._sky_in_q.put(None)
            self._sky_thread.join()
        if self._montage_thread is not None:
            self._montage_queue.put(None)
            self._montage_thread.join()
        return super().destroy_node()

    def _publish_image(self, array, encoding, source_message, publisher):
        array = np.ascontiguousarray(array)
        message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
        message.header.stamp = source_message.header.stamp
        message.header.frame_id = source_message.header.frame_id
        publisher.publish(message)


def _parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="ROS 2 Depth Anything V2 TensorRT relative-depth node")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                        help=f"TensorRT engine path (default: {DEFAULT_ENGINE})")
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=SKY_REFINE_ENABLE,
        help="enable confidence-weighted sky-mask refinement (default: enabled)",
    )
    return parser.parse_known_args(args)


def main(args=None):
    cli_args, ros_args = _parse_arguments(args)
    engine_path = cli_args.engine.expanduser().resolve()
    rclpy.init(args=ros_args)
    node = DepthAnythingV2TensorRTNode(
        engine_path=engine_path,
        refine_enabled=cli_args.refine,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
