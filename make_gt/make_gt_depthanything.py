#!/usr/bin/env python3
"""
Metric GT for SML training -- DA-V2 on the LEFT rect image, anchored by the
RealSense depth warped from the INFRA1 (IR) frame straight into the left frame.

Why infra1, not color
---------------------
The Kalibr calibration is BETWEEN infra1 (cam0) and stereo-left (cam1), and its
`T_cn_cnm1` for cam1 IS T_{L<-I}. RealSense `depth/image_rect_raw` is native
depth in the infra1 optical frame (not reprojected to color). So warping it into
the left frame is a SINGLE, well-calibrated hop:

    P_I = backproject(depth_I, K_I)           # 3D in infra1 frame
    P_L = T_{L<-I} @ P_I                       # 3D in stereo-left frame
    (u,v) = project(P_L, cam1 K + radtan)      # left pixel

No color camera, no RealSense factory depth->color extrinsic, no double warp --
which is what was silently dropping most anchors before.

Pipeline
--------
  1. DA-V2 on the LEFT rect image        -> relative disparity (left grid).
  2. Warp RealSense infra1 depth -> left -> dense metric anchors (left grid).
  3. Robust affine fit DA(left) -> metric -> dense metric depth (left frame).
  4. Apply an optional deploy FOV mask; without one, use the whole image.
  5. Export (LightStereo disparity, GT) training pair.
  6. Per-step montage + up/down compare of the (GT, disparity) pair.

Deps: rosbags, numpy, opencv-python, torch, transformers, pillow
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

try:
    from .correct_matrix_direction import load_d455_d435, load_left_d455
except ImportError:  # direct execution: python3 sml/make_gt/make_gt_depthanything.py
    from correct_matrix_direction import load_d455_d435, load_left_d455

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
GT_CONFIG_PATH = "gt_config.yaml"

# --------------------------------------------------------------------------- #
# Calibration (from calib_ir_...-camchain-0.yaml)
#   cam0 = infra1 (/camera/camera/infra1/image_rect_raw), 640x480 pinhole+radtan
#   cam1 = stereo-left (/stereo_1_0/left/image_rect),      320x320 pinhole+radtan
#   cam1.T_cn_cnm1 = T_{L<-I}  (maps a point in infra1 -> stereo-left)
# --------------------------------------------------------------------------- #
CAM0_PROJ = np.array([391.1833053665675, 391.2023258821923,
                      315.333377935706, 240.48835472327764])          # infra1 fx fy cx cy
CAM1_PROJ = np.array([167.07658738687104, 158.31284216746292,
                      138.2772130077489, 156.41669349148265])          # left  fx fy cx cy
CAM1_DIST = np.array([-0.032352588466769784, -0.0244944119435387,
                      -0.011755132756719569, -0.018714907886535945])   # left  k1 k2 p1 p2

T_LI = np.array([
    [0.9877923929746802, -0.03442690680274639, 0.15192424582453154, -0.021341394350870197],
    [0.036278084304550484, 0.999297249809828, -0.009429057307547176, 0.05351095296832991],
    [-0.15149286775472173, 0.014825471679245007, 0.9883471639100003, -0.011077487760199874],
    [0.0, 0.0, 0.0, 1.0],
])  # T_{L<-I}   (I = D455 infra1/depth)

# TOPIC_DEPTH = "/camera/camera/depth/image_rect_raw"          # depth in infra1 frame
# TOPIC_DEPTH_INFO = "/camera/camera/depth/camera_info"

TOPIC_DEPTH = "/d455/d455_node/depth/image_rect_raw"
TOPIC_DEPTH_INFO = "/d455/d455_node/depth/camera_info"

TOPIC_LEFT = "/stereo_1_0/left/image_rect"
TOPIC_DISP = "/stereo_1_0/disparity"


# --------------------------------------------------------------------------- #
# Second RealSense (D435), mounted BELOW the D455.
#
#   fisheye 0 -- 1
#   D455  (its infra1 depth = current step 2)
#   D435  (its color-aligned depth = NEW step 2b)
#
# The provided calibration is COLOR<->COLOR (Kalibr):
#   cam0 = /d435/.../color   cam1 = /d455/.../color
#   baseline T_1_0 = T_{D455color <- D435color}.
#
# To bring D435 depth into the LEFT frame we chain:
#   T_{L<-D435c} = T_{L<-I455} . T_{I455<-C455} . T_{C455<-D435c}
#                  \___T_LI___/  \_D455 factory_/  \__new calib__/
#
# T_{I455<-C455} is the D455 factory Depth->Color extrinsic (rs-R/rs-t you used
# earlier); it bridges the fact that T_LI lands in D455 *infra1* while the new
# calibration starts from D455 *color*. Override via --d455-d2c-R/--d455-d2c-t
# if your device differs.
# --------------------------------------------------------------------------- #
def quat_xyzw_to_R(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_T(R, t):
    M = np.eye(4); M[:3, :3] = np.asarray(R); M[:3, 3] = np.asarray(t).reshape(3)
    return M


# D435 color intrinsics + distortion (from the attached Kalibr result, cam0)
D435_COLOR_PROJ = np.array([890.53017846, 892.8789643, 634.55621986, 361.46349414])
D435_COLOR_DIST = np.array([0.08851802, -0.15578615, -0.00301015, -0.00175584])  # radtan

# T_{D455color <- D435color}  (Kalibr baseline T_1_0: q xyzw, t)
_Q_C455_C435 = np.array([-0.01405267, -0.025424, -0.01430282, 0.99947565])
_T_C455_C435 = np.array([-0.02967851, 0.07248771, -0.01710743])
T_C455_C435 = make_T(quat_xyzw_to_R(_Q_C455_C435), _T_C455_C435)

# --------------------------------------------------------------------------- #
# 2026-08-27 direct Infra1-to-Infra1 Kalibr calibration.
#
# Frame-name convention used below:
#   I455 = D455 Infra1 / depth optical frame
#   I435 = D435 Infra1 / depth optical frame
#   C455 = D455 color optical frame
#   C435 = D435 color optical frame
#   L    = stereo-left optical frame
#
# The camchain used cam0=/d455/.../infra1/image_rect_raw and
# cam1=/d435/.../infra1/image_rect_raw.  Kalibr's cam1.T_cn_cnm1 therefore is
# T_{I435<-I455}: it maps a point from D455 Infra1 into D435 Infra1.
# It CANNOT be pasted directly into T_C455_C435 above, because that variable is
# T_{C455<-C435} and its source/target frames are the two COLOR cameras.
#
# For raw D435 depth the shortest and preferred chain is:
#   P_L = T_{L<-I455} @ inv(T_{I435<-I455}) @ P_I435
#       = T_LI         @ inv(T_I435_I455)    @ P_I435
# This avoids unnecessary Infra1->Color->Color->Infra1 conversions.
#
# If a color-frame transform is ever required, derive it using the factory
# depth-to-color extrinsics instead of using this matrix directly:
#   T_{C455<-C435} = T_{C455<-I455}
#                    @ inv(T_{I435<-I455})
#                    @ inv(T_{C435<-I435})
# With the factory D2C values below, the equivalent color-to-color calibration
# is approximately:
#   q_xyzw = [-0.165177722499, 0.032599476186,
#              0.005731316168, 0.985708245962]
#   t_xyz  = [-0.013958810419, 0.092580924484, -0.009892141436]
#
# This calibration only updates D435<->D455.  If the D455 moved relative to the
# stereo-left camera, T_LI above must be recalibrated separately.
T_I435_I455 = np.array([
    [0.9977874435047719, 0.006856618065229324,
     -0.06613020771872238, -0.061219490856716345],
    [-0.028036512738727905, 0.9452950354703613,
     -0.3250096150401424, -0.08852231722164512],
    [0.060284090253274504, 0.3261445733164941,
     0.943395752459454, -0.023948287617004543],
    [0.0, 0.0, 0.0, 1.0],
])  # T_{I435<-I455}; invert it when mapping D435 raw depth toward D455/left

# D455 factory Depth(infra1)->Color extrinsic  T_{C455 <- I455}  (override via CLI)
D455_D2C_R = np.array([
    [0.9999980330467224, -0.0011719099711626768, 0.0015828418545424938],
    [0.0011774318991228938, 0.9999932050704956, -0.003492199582979083],
    [-0.0015787385636940598, 0.0034940566401928663, 0.9999926686286926],
])
D455_D2C_t = np.array([-0.05906914919614792, 0.0005169452633708715, -0.0005152876838110387])

# D435 factory Depth(infra1)->Color extrinsic  T_{C435 <- I435}  (override via CLI).
# Only needed when feeding the D435 RAW depth (infra1 frame) instead of its
# color-aligned depth.
D435_D2C_R = np.array([
    [0.999968945980072, -0.007811499759554863, -0.001052754931151867],
    [0.00780802546069026, 0.99996417760849, -0.0032648355700075626],
    [0.0010782205499708652, 0.0032565142028033733, 0.9999940991401672],
])
D435_D2C_t = np.array([0.014820273034274578, -0.00016969414718914777, 0.00032040890073403716])

# D435 depth topics
TOPIC_D435_DEPTH = "/d435/d435_node/aligned_depth_to_color/image_raw"   # color frame
TOPIC_D435_DEPTH_RAW = "/d435/d435_node/depth/image_rect_raw"           # infra1 frame


def compose_T_L_from_D435color(T_c455_i455):
    """T_{L<-D435color} = T_LI . inv(T_{C455<-I455}) . T_{C455<-D435c}."""
    T_L_c455 = T_LI @ np.linalg.inv(T_c455_i455)      # L <- D455 color
    return T_L_c455 @ T_C455_C435                     # L <- D435 color


def compose_T_L_from_D435infra1(T_c455_i455, T_c435_i435):
    """T_{L<-D435infra1} = T_{L<-D435color} . T_{C435<-I435}.

    Use this when feeding the D435 RAW depth (native infra1 frame, pinhole) --
    consistent with how the D455 path uses rectified infra1 depth, and it avoids
    the color-aligned resampling + color-lens distortion.
    """
    return compose_T_L_from_D435color(T_c455_i455) @ T_c435_i435


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def stamp_to_sec(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def image_to_numpy(msg):
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("16uc1", "mono16"):
        return np.ascontiguousarray(buf.view("<u2").reshape(h, step // 2)[:, :w])
    if enc in ("mono8", "8uc1"):
        return np.ascontiguousarray(buf.reshape(h, step)[:, :w])
    if enc == "32fc1":
        return np.ascontiguousarray(buf.view("<f4").reshape(h, step // 4)[:, :w])
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(arr[..., ::-1] if enc == "rgb8" else arr)
    if enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[..., :3]
        return np.ascontiguousarray(arr[..., ::-1] if enc == "rgba8" else arr)
    raise ValueError(f"unhandled encoding {msg.encoding}")


def load_topics(bagpath, topics):
    out = {t: [] for t in topics}
    with AnyReader([Path(bagpath)], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            out[conn.topic].append((stamp_to_sec(msg), msg))
    for t in out:
        out[t].sort(key=lambda x: x[0])
    return out


def nearest(sorted_list, t, tol):
    if not sorted_list:
        return None
    times = np.array([x[0] for x in sorted_list])
    i = int(np.argmin(np.abs(times - t)))
    return sorted_list[i] if abs(times[i] - t) <= tol else None


def load_da_v2(model_id, device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return proc, model


def load_deploy_mask(path, out_hw):
    m = np.load(path)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(bool)
    if m.shape != out_hw:
        m = cv2.resize(m.astype(np.uint8), (out_hw[1], out_hw[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
    return m


def load_gt_config(path):
    """Load one camera/RealSense calibration set for GT generation."""
    if not path:
        return {}, None
    import yaml
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, cfg_path


def apply_gt_calibration(cfg):
    """Apply calibration/topic values while preserving legacy defaults."""
    global CAM0_PROJ, CAM1_PROJ, CAM1_DIST, T_LI, T_I435_I455
    global TOPIC_DEPTH, TOPIC_DEPTH_INFO, TOPIC_LEFT, TOPIC_DISP

    topics = cfg.get("topics", {})
    target = cfg.get("target", {})
    d455 = cfg.get("d455", {})
    transforms = cfg.get("transforms", {})

    TOPIC_DEPTH = topics.get("d455_depth", TOPIC_DEPTH)
    TOPIC_DEPTH_INFO = topics.get("d455_camera_info", TOPIC_DEPTH_INFO)
    TOPIC_LEFT = topics.get("left", TOPIC_LEFT)
    TOPIC_DISP = topics.get("disparity", TOPIC_DISP)

    if d455.get("intrinsics") is not None:
        CAM0_PROJ = np.asarray(d455["intrinsics"], dtype=float)
    if target.get("intrinsics") is not None:
        CAM1_PROJ = np.asarray(target["intrinsics"], dtype=float)
    if target.get("distortion") is not None:
        CAM1_DIST = np.asarray(target["distortion"], dtype=float)
    if transforms.get("target_from_d455_depth") is not None:
        T_LI = np.asarray(transforms["target_from_d455_depth"], dtype=float)
    if transforms.get("d435_depth_from_d455_depth") is not None:
        T_I435_I455 = np.asarray(
            transforms["d435_depth_from_d455_depth"], dtype=float)

    for name, value, shape in (
        ("d455.intrinsics", CAM0_PROJ, (4,)),
        ("target.intrinsics", CAM1_PROJ, (4,)),
        ("target.distortion", CAM1_DIST, (4,)),
        ("transforms.target_from_d455_depth", T_LI, (4, 4)),
        ("transforms.d435_depth_from_d455_depth", T_I435_I455, (4, 4)),
    ):
        if np.asarray(value).shape != shape:
            raise SystemExit(
                f"invalid {name} shape: expected {shape}, got {np.asarray(value).shape}"
            )


def config_path(value, cfg_path):
    """Resolve paths in a GT config relative to that YAML file."""
    if not value or cfg_path is None:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (cfg_path.parent / path).resolve())


# =========================================================================== #
# Sky segmentation (NCNN)  -- filters sky out of the affine fit and exports a
# sky mask so train_sml_global.py can label sky as a fixed far depth.
# =========================================================================== #
def _bias(x, b=0.8):
    return x / (((1.0 / b) - 2.0) * (1.0 - x) + 1.0)


def probability_to_confidence(prob, low=0.3, high=0.5, bias_b=0.8, eps=0.01):
    conf = np.full_like(prob, eps, dtype=np.float32)
    low_mask = prob < low
    high_mask = prob > high
    conf[low_mask] = np.maximum(
        _bias((low - prob[low_mask]) / low, bias_b), eps)
    conf[high_mask] = np.maximum(
        _bias((prob[high_mask] - high) / (1.0 - high), bias_b), eps)
    return conf


def refine_sky_prob(prob, gray, radius=24, eps=1e-3, low=0.3, high=0.5,
                    bias_b=0.8, do_bilateral=True):
    """Confidence-weighted guided-filter refinement of sky probability."""
    probability = prob.astype(np.float32)
    guide = gray.astype(np.float32)
    if guide.max() > 1.5:
        guide /= 255.0
    weight = probability_to_confidence(
        probability, low=low, high=high, bias_b=bias_b)
    kernel = (2 * radius + 1, 2 * radius + 1)

    def box(value):
        return cv2.boxFilter(value, -1, kernel, normalize=False,
                             borderType=cv2.BORDER_REPLICATE)

    total_weight = box(weight) + 1e-8
    mean_guide = box(weight * guide) / total_weight
    mean_prob = box(weight * probability) / total_weight
    variance = box(weight * guide * guide) / total_weight - mean_guide**2
    covariance = box(weight * guide * probability) / total_weight
    covariance -= mean_guide * mean_prob
    slope = covariance / (variance + eps)
    offset = mean_prob - slope * mean_guide
    refined = box(weight * slope) / total_weight * guide
    refined += box(weight * offset) / total_weight
    refined = np.clip(refined, 0.0, 1.0)
    if do_bilateral:
        refined = cv2.bilateralFilter(refined, 0, 0.08, 8)
    return refined


class SkySegmenter:
    """
    Thin wrapper around an NCNN sky-segmentation model (e.g. the EGE-UNet model
    from github.com/kccccck/sky-segmentation).

    The exact blob names / input size / normalization depend on how the .param
    was exported, so everything is configurable. Defaults follow the common
    EGE / ImageNet-normalized convention; adjust to match your model:

      * --sky-input-name / --sky-output-name : blob names in the .param
      * --sky-size                           : network input side (square)
      * --sky-mean / --sky-norm              : substract_mean_normalize args
      * --sky-sigmoid                        : apply sigmoid to logits
      * --sky-thresh                         : sky if prob > thresh
      * --sky-invert                         : flip if your model outputs
                                               foreground=0/sky=1 vs the reverse
    """
    def __init__(self, param, bin, size=320, input_name="in0", output_name="out0",
                 mean=(123.675, 116.28, 103.53), norm=(0.01712, 0.01751, 0.01743),
                 sigmoid=True, thresh=0.5, invert=False, use_gpu=False,
                 refine=False, refine_radius=24, refine_eps=1e-3,
                 refine_low=0.3, refine_high=0.5, refine_bias=0.8,
                 refine_bilateral=True):
        import ncnn
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = bool(use_gpu)
        self.net.load_param(param)
        self.net.load_model(bin)
        self.size = int(size)
        self.input_name = input_name
        self.output_name = output_name
        self.mean = list(mean)
        self.norm = list(norm)
        self.sigmoid = sigmoid
        self.thresh = float(thresh)
        self.invert = invert
        self.refine = bool(refine)
        self.refine_radius = int(refine_radius)
        self.refine_eps = float(refine_eps)
        self.refine_low = float(refine_low)
        self.refine_high = float(refine_high)
        self.refine_bias = float(refine_bias)
        self.refine_bilateral = bool(refine_bilateral)

    def prob(self, bgr):
        """Return HxW float sky probability at the input image resolution."""
        h0, w0 = bgr.shape[:2]
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        mat_in = self.ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(bgr), self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            w0, h0, self.size, self.size)
        mat_in.substract_mean_normalize(self.mean, self.norm)
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        _, mat_out = ex.extract(self.output_name)
        out = np.array(mat_out)                      # (C,H,W) or (H,W)
        if out.ndim == 3:
            p = out[0] if out.shape[0] == 1 else out[-1]   # 1ch sigmoid, or last logit
        else:
            p = out
        if self.sigmoid:
            p = 1.0 / (1.0 + np.exp(-p))
        p = cv2.resize(p.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)
        return 1.0 - p if self.invert else p

    def mask(self, bgr):
        probability = self.prob(bgr)
        if self.refine:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            probability = refine_sky_prob(
                probability, gray, radius=self.refine_radius,
                eps=self.refine_eps, low=self.refine_low,
                high=self.refine_high, bias_b=self.refine_bias,
                do_bilateral=self.refine_bilateral)
        return probability > self.thresh


def heuristic_sky_mask(left_img, disk):
    """
    Fallback sky detector when no NCNN model is provided. Sky in a forward
    fisheye view is bright, low-texture, and toward the top of the disk.
    Coarse, but better than nothing for filtering fit anchors.
    """
    g = left_img if left_img.ndim == 2 else cv2.cvtColor(left_img[..., :3], cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    bright = g > (0.6 * 255)
    grad = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    smooth = np.abs(grad) < 12.0
    H = g.shape[0]
    yy = np.arange(H)[:, None] < int(0.6 * H)          # upper 60% of the frame
    sky = bright & smooth & yy & disk
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return sky > 0


# =========================================================================== #
# STEP 1 - DA-V2 relative disparity on the LEFT rect image
# =========================================================================== #
def step1_da_on_left(proc, model, left_img, device):
    import torch, torch.nn.functional as F
    left_bgr = left_img if left_img.ndim == 3 else cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = F.interpolate(pred[:, None], size=left_img.shape[:2], mode="bicubic",
                         align_corners=False)[0, 0]
    return pred.float().cpu().numpy()


# =========================================================================== #
# STEP 2 - warp RealSense INFRA1 depth INTO the left frame (single Kalibr hop)
# =========================================================================== #
def step2_ir_depth_to_L(depth_I_m, K_I, T_L_I, cam1_proj, cam1_dist, out_hw,
                        splat=True, src_dist=None):
    """
    Warp a metric depth map from a source pinhole camera into the LEFT frame.

    depth_I_m : (h,w) metric depth (m) in the source frame
    K_I       : (fx,fy,cx,cy) source intrinsics
    T_L_I     : 4x4  T_{L<-source}
    src_dist  : optional radtan (k1,k2,p1,p2) of the SOURCE camera. If given,
                pixels are back-projected through cv2.undistortPoints (needed for
                the D435 color-aligned depth, which carries color-lens distortion;
                the D455 rectified infra1 depth passes src_dist=None).
    returns   : (H,W) metric depth in the left frame, NaN where empty
    """
    fx, fy, cx, cy = K_I
    h, w = depth_I_m.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_I_m
    m = np.isfinite(z) & (z > 0)
    u = uu[m].astype(np.float64); v = vv[m].astype(np.float64); z = z[m].astype(np.float64)
    if src_dist is not None:
        Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        pts = np.stack([u, v], axis=1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, Kmat, np.asarray(src_dist, np.float64)).reshape(-1, 2)
        X = und[:, 0] * z; Y = und[:, 1] * z            # undistorted normalized rays * Z
    else:
        X = (u - cx) / fx * z; Y = (v - cy) / fy * z
    P = np.stack([X, Y, z, np.ones_like(z)], axis=1)
    P1 = (T_L_I @ P.T).T[:, :3]                              # source -> left
    Z1 = P1[:, 2]; fr = Z1 > 1e-6
    P1, Z1 = P1[fr], Z1[fr]
    xn = P1[:, 0] / Z1; yn = P1[:, 1] / Z1
    k1, k2, p1, p2 = cam1_dist
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    fx1, fy1, cx1, cy1 = cam1_proj
    uf = fx1 * xd + cx1
    vf = fy1 * yd + cy1
    H, W = out_hw
    out = np.full((H, W), np.inf, np.float32)               # inf so z-buffer min works
    # optional 2x2 splat to close sub-pixel gaps when downsampling 640->320
    offs = [(0, 0), (0, 1), (1, 0), (1, 1)] if splat else [(0, 0)]
    for du, dv in offs:
        ui = np.floor(uf + du).astype(np.int64)
        vi = np.floor(vf + dv).astype(np.int64)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        np.minimum.at(out, (vi[inb], ui[inb]), Z1[inb].astype(np.float32))
    out[~np.isfinite(out)] = np.nan
    return out


def merge_depth_L(primary, secondary, mode="fill"):
    """
    Merge two left-frame depth maps.
      fill : keep `primary` where valid, fill its holes with `secondary`
             (D455 is better-calibrated, so it wins; D435 adds coverage below).
      min  : nearer of the two wherever both are valid (occlusion-safe).
    """
    if secondary is None:
        return primary
    out = primary.copy()
    if mode == "min":
        both = np.isfinite(primary) & np.isfinite(secondary)
        out[both] = np.minimum(primary[both], secondary[both])
        only2 = ~np.isfinite(primary) & np.isfinite(secondary)
        out[only2] = secondary[only2]
    else:  # fill
        hole = ~np.isfinite(primary) & np.isfinite(secondary)
        out[hole] = secondary[hole]
    return out


# =========================================================================== #
# STEP 3 - robust affine fit  DA(left) -> metric depth (left frame)
# =========================================================================== #
def robust_affine_invdepth(rel, inv_gt, valid, num=3000, ransac_iters=200,
                           iters=3, k=2.5, rng=None):
    rng = rng or np.random.default_rng(0)
    ys, xs = np.where(valid)
    if len(xs) < 50:
        return None
    if len(xs) > num:
        sel = rng.choice(len(xs), num, replace=False)
        ys, xs = ys[sel], xs[sel]
    x = rel[ys, xs].astype(np.float64)
    y = inv_gt[ys, xs].astype(np.float64)
    tau = 0.3 * (1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9)
    bs, bt, bi, N = 1.0, 0.0, -1, len(x)
    for _ in range(ransac_iters):
        i, j = rng.integers(0, N, size=2)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        s = (y[i] - y[j]) / (x[i] - x[j]); t = y[i] - s * x[i]
        inl = int((np.abs(y - (s * x + t)) < tau).sum())
        if inl > bi:
            bs, bt, bi = s, t, inl
    s, t = bs, bt
    keep = np.abs(y - (s * x + t)) < tau
    for _ in range(iters):
        if keep.sum() < 50:
            break
        A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        s, t = float(sol[0]), float(sol[1])
        res = y - (s * x + t); med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med)) + 1e-9
        keep = np.abs(res - med) < k * mad
    return s, t, float(keep.mean())


def step3_fit_metric_L(da_L, rs_depth_L, anchor_valid, da_metric=False,
                       max_depth=30.0):
    """
    Fit DA(left) -> metric using the RS anchors, then convert the WHOLE da_L to
    metric depth. Pixels the fit pushes beyond `max_depth` (or to <=0) are marked
    NaN, NOT clipped -- clipping to 1e-3 is what produced the 999.999 m sentinel
    that corrupted training. Keep this gating.
    """
    if da_metric:
        v = anchor_valid & np.isfinite(da_L) & (da_L > 0)
        if v.sum() < 50:
            return None, {}
        ratio = float(np.median(rs_depth_L[v] / da_L[v]))
        out = (da_L * ratio).astype(np.float32)
        out[~np.isfinite(out) | (out <= 0) | (out > max_depth)] = np.nan
        return out, {"mode": "metric", "ratio": ratio, "inl": 1.0}

    valid_anchor = anchor_valid & np.isfinite(rs_depth_L) & (rs_depth_L > 0)
    inv_gt = np.zeros_like(rs_depth_L)
    inv_gt[valid_anchor] = 1.0 / rs_depth_L[valid_anchor]
    fit = robust_affine_invdepth(da_L, inv_gt, valid_anchor)
    if fit is None:
        return None, {}
    s, t, inl = fit
    ga_inv = s * da_L + t
    thr = 1.0 / max_depth
    good = np.isfinite(ga_inv) & (ga_inv > thr)
    depth_L = np.full_like(da_L, np.nan, dtype=np.float32)
    depth_L[good] = (1.0 / ga_inv[good]).astype(np.float32)
    return depth_L, {"mode": "affine", "s": s, "t": t, "inl": inl,
                     "coverage": float(good.mean())}


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _depth_bgr(depth, mask, vmin, vmax):
    v = mask & np.isfinite(depth)
    norm = np.zeros(depth.shape, np.float32)
    norm[v] = np.clip((depth[v] - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~v] = 30
    return col


def _rel_bgr(x, mask=None):
    v = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    norm = np.zeros(x.shape, np.float32)
    if v.any():
        lo, hi = np.percentile(x[v], [2, 98])
        norm[v] = np.clip((x[v] - lo) / max(hi - lo, 1e-6), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    col[~v] = 30
    return col


def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _tile(img_bgr, title, tile, vmin=None, vmax=None):
    t = cv2.resize(img_bgr, (tile, tile), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(t, (0, 0), (tile - 1, 20), (0, 0, 0), -1)
    cv2.putText(t, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    if vmin is not None and vmax is not None:
        bw = 14
        bar = cv2.applyColorMap(
            np.repeat(np.linspace(255, 0, tile, dtype=np.uint8)[:, None], bw, 1),
            cv2.COLORMAP_TURBO)
        lab = np.full((tile, 34, 3), 20, np.uint8)
        for frac, val in [(0.04, vmax), (0.5, (vmin + vmax) / 2), (0.96, vmin)]:
            cv2.putText(lab, f"{val:.1f}", (2, int(tile * frac) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        t = np.hstack([t, bar, lab])
    return t


def build_montage(left_img, da_L, depth_I, rs_depth_L, anchor_valid, depth_L, info,
                  fov_mask, gt_valid, disp, tile, dmin, dmax, sky_mask=None):
    left_bgr = _to_bgr(left_img)
    da_rel = _rel_bgr(da_L)
    depthI_col = _depth_bgr(depth_I, np.isfinite(depth_I) & (depth_I > 0), dmin, dmax)
    rs_anchor = _depth_bgr(rs_depth_L, anchor_valid, dmin, dmax)
    dL = _depth_bgr(depth_L, np.isfinite(depth_L), dmin, dmax)
    fov = _to_bgr((fov_mask.astype(np.uint8) * 255))
    gtm = _depth_bgr(depth_L, gt_valid, dmin, dmax)
    dsp = _rel_bgr(disp, np.isfinite(disp) & (disp > 0))
    fit_txt = (f"3 out metric(L) s={info.get('s',0):.2f} inl={info.get('inl',0)*100:.0f}%"
               if info.get("mode") == "affine"
               else f"3 out metric(L) x{info.get('ratio',1):.2f}")

    # sky overlay on the left image (cyan = sky)
    if sky_mask is not None and sky_mask.any():
        sky_over = left_bgr.copy()
        sky_over[sky_mask] = (0.4 * sky_over[sky_mask]
                              + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
        sky_txt = f"sky seg ({sky_mask.mean()*100:.0f}%)"
    else:
        sky_over = left_bgr.copy()
        sky_txt = "sky seg (0%)"

    row1 = np.hstack([
        _tile(left_bgr,    "1 in left rect", tile),
        _tile(da_rel,      "1 out DA disp (rel)", tile),
        _tile(depthI_col,  "2 in RS depth (infra1)", tile, dmin, dmax),
        _tile(rs_anchor,   f"2 out RS depth->L ({np.isfinite(rs_depth_L).mean()*100:.0f}%)",
              tile, dmin, dmax),
    ])
    row2 = np.hstack([
        _tile(dL,  fit_txt, tile, dmin, dmax),
        _tile(sky_over, sky_txt, tile),
        _tile(gtm, f"4 out GT masked ({gt_valid.mean()*100:.0f}%)", tile, dmin, dmax),
        _tile(dsp, "5 LightStereo disp", tile),
    ])
    up = _tile(gtm, "(5) GT masked", tile, dmin, dmax)
    down = _tile(dsp, "(5) LightStereo disp", tile)
    w = max(up.shape[1], down.shape[1])
    padw = lambda t: np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=20)
    pair = np.vstack([padw(up), padw(down)])
    W = max(row1.shape[1], row2.shape[1], pair.shape[1])
    padr = lambda r: np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=20)
    return np.vstack([padr(row1), padr(row2), padr(pair)])


def fit_is_suspect(depth_L, gt_valid, info, min_spread, min_scale, min_inlier):
    """Flag near-constant or poorly fitted global-alignment outputs."""
    reasons = []
    values = depth_L[gt_valid & np.isfinite(depth_L)]
    if values.size < 50:
        return True, ["too_few_valid"]
    p10, p90 = np.percentile(values, [10, 90])
    spread = p90 / max(p10, 1e-6)
    if spread < min_spread:
        reasons.append(f"spread={spread:.2f}<{min_spread}")
    if info.get("mode") == "affine":
        if abs(info.get("s", 0.0)) < min_scale:
            reasons.append(f"s={info.get('s', 0.0):.4f}<{min_scale}")
        if info.get("inl", 1.0) < min_inlier:
            reasons.append(f"inl={info.get('inl', 0.0):.2f}<{min_inlier}")
    return bool(reasons), reasons


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global CAM0_PROJ, CAM1_PROJ, CAM1_DIST, T_LI

    default_config = str(Path(__file__).with_name(GT_CONFIG_PATH))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gt-config", default=default_config)
    pre_args, _ = pre.parse_known_args()
    gt_cfg, gt_cfg_path = load_gt_config(pre_args.gt_config)
    # Preserve compatibility with older GT configs containing explicit matrices.
    apply_gt_calibration(gt_cfg)
    io_cfg = gt_cfg.get("io", {})
    topics_cfg = gt_cfg.get("topics", {})
    calibration_cfg = gt_cfg.get("calibration", {})
    mask_cfg = gt_cfg.get("mask", {})
    runtime_cfg = gt_cfg.get("runtime", {})
    da_cfg = gt_cfg.get("da", {})
    sky_cfg = gt_cfg.get("sky", {})
    qc_cfg = gt_cfg.get("qc", {})
    d435_cfg = gt_cfg.get("d435", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-config", default=pre_args.gt_config,
                    help="GT runtime config; not a Kalibr camchain")
    ap.add_argument("--bag", default=config_path(io_cfg.get("bag"), gt_cfg_path))
    ap.add_argument("--vis-dir",
                    default=config_path(io_cfg.get("vis_dir", "./gt_vis"), gt_cfg_path))
    ap.add_argument("--export-dir",
                    default=config_path(io_cfg.get("export_dir"), gt_cfg_path))
    ap.add_argument("--max-pairs", type=int, default=io_cfg.get("max_pairs", 1000))
    ap.add_argument("--tile", type=int, default=io_cfg.get("tile", 300))

    # Topics: CLI overrides GT config, which overrides legacy defaults.
    ap.add_argument("--left-topic", default=topics_cfg.get("left", TOPIC_LEFT))
    ap.add_argument("--disp-topic", default=topics_cfg.get("disparity", TOPIC_DISP))
    ap.add_argument("--d455-depth",
                    default=topics_cfg.get("d455_depth", TOPIC_DEPTH))
    ap.add_argument("--d455-info",
                    default=topics_cfg.get("d455_camera_info", TOPIC_DEPTH_INFO))
    ap.add_argument("--d435-depth",
                    default=topics_cfg.get("d435_depth", d435_cfg.get("depth_topic")))
    ap.add_argument("--d435-info", default=topics_cfg.get(
        "d435_camera_info", d435_cfg.get("camera_info_topic")))

    # Raw, unmodified Kalibr outputs. Both camchain YAML and results TXT work.
    ap.add_argument("--left-calib", default=config_path(
        calibration_cfg.get("left_d455"), gt_cfg_path))
    ap.add_argument("--d435-calib", default=config_path(
        calibration_cfg.get("d455_d435"), gt_cfg_path))
    ap.add_argument("--left-proj", type=float, nargs=4, default=None)
    ap.add_argument("--left-dist", type=float, nargs=4, default=None)
    ap.add_argument("--d455-proj", type=float, nargs=4,
                    default=calibration_cfg.get("d455_intrinsics"))
    ap.add_argument("--d435-proj", type=float, nargs=4,
                    default=calibration_cfg.get(
                        "d435_intrinsics", d435_cfg.get("intrinsics")))
    ap.add_argument("--d435-source",
                    choices=["color", "infra1", "kalibr_infra1"],
                    default=calibration_cfg.get(
                        "d435_source", d435_cfg.get("source", "color")))

    ap.add_argument(
        "--deploy-mask",
        default=config_path(mask_cfg.get(
            "deploy_mask", gt_cfg.get("deploy_mask")), gt_cfg_path),
        help="optional .npy FOV mask; without it the whole target image is valid",
    )
    ap.add_argument("--no-deploy-mask", dest="deploy_mask", action="store_const",
                    const=None, help="override the GT config and use the whole image")

    ap.add_argument("--da-model", default=da_cfg.get(
        "model", "depth-anything/Depth-Anything-V2-Small-hf"))
    ap.add_argument("--da-metric", dest="da_metric", action="store_true",
                    default=da_cfg.get("metric", False))
    ap.add_argument("--no-da-metric", dest="da_metric", action="store_false")
    ap.add_argument("--depth-scale", type=float, default=calibration_cfg.get(
        "d455_depth_scale", runtime_cfg.get("depth_scale", 0.001)))
    ap.add_argument("--d435-depth-scale", type=float,
                    default=calibration_cfg.get(
                        "d435_depth_scale", d435_cfg.get("depth_scale", 0.001)))
    ap.add_argument("--sync-tol", type=float, default=runtime_cfg.get("sync_tol", 0.03))
    ap.add_argument("--depth-time-offset", type=float,
                    default=runtime_cfg.get("depth_time_offset", 0.0))
    ap.add_argument("--d435-time-offset", type=float,
                    default=runtime_cfg.get(
                        "d435_time_offset", d435_cfg.get("time_offset", 0.0)))
    ap.add_argument("--dmin", type=float, default=runtime_cfg.get("dmin", 0.2))
    ap.add_argument("--dmax", type=float, default=runtime_cfg.get("dmax", 15.0))
    ap.add_argument("--max-fit-depth", type=float,
                    default=runtime_cfg.get("max_fit_depth", 10.0))
    ap.add_argument("--gt-max-depth", type=float,
                    default=runtime_cfg.get("gt_max_depth", 20.0))
    ap.add_argument("--d435-merge", choices=["fill", "min"],
                    default=runtime_cfg.get(
                        "d435_merge", d435_cfg.get("merge", "fill")))
    ap.add_argument("--no-splat", dest="no_splat", action="store_true",
                    default=not runtime_cfg.get("splat", True),
                    help="disable 2x2 splat (leave sub-pixel holes in the warp)")
    ap.add_argument("--splat", dest="no_splat", action="store_false")

    # ---- sky segmentation (NCNN) ----
    ap.add_argument("--sky", dest="sky_enabled", action="store_true",
                    default=sky_cfg.get("enabled", False))
    ap.add_argument("--no-sky", dest="sky_enabled", action="store_false")
    ap.add_argument("--sky-param", default=config_path(sky_cfg.get("param"), gt_cfg_path))
    ap.add_argument("--sky-bin", default=config_path(sky_cfg.get("bin"), gt_cfg_path))
    ap.add_argument("--sky-size", type=int, default=sky_cfg.get("size", 320))
    ap.add_argument("--sky-input-name", default=sky_cfg.get("input_name", "in0"))
    ap.add_argument("--sky-output-name", default=sky_cfg.get("output_name", "out0"))
    ap.add_argument("--sky-mean", type=float, nargs=3,
                    default=sky_cfg.get("mean", [123.675, 116.28, 103.53]))
    ap.add_argument("--sky-norm", type=float, nargs=3,
                    default=sky_cfg.get("norm", [0.01712, 0.01751, 0.01743]))
    ap.add_argument("--sky-no-sigmoid", dest="sky_no_sigmoid", action="store_true",
                    default=not sky_cfg.get("apply_sigmoid", True),
                    help="model already outputs probabilities (skip sigmoid)")
    ap.add_argument("--sky-sigmoid", dest="sky_no_sigmoid", action="store_false")
    ap.add_argument("--sky-thresh", type=float, default=sky_cfg.get("threshold", 0.5))
    ap.add_argument("--sky-invert", dest="sky_invert", action="store_true",
                    default=sky_cfg.get("invert", False))
    ap.add_argument("--no-sky-invert", dest="sky_invert", action="store_false")
    ap.add_argument("--sky-heuristic", dest="sky_heuristic", action="store_true",
                    default=sky_cfg.get("heuristic", False))
    ap.add_argument("--no-sky-heuristic", dest="sky_heuristic", action="store_false")
    ap.add_argument("--sky-gpu", dest="sky_gpu", action="store_true",
                    default=sky_cfg.get("gpu", False))
    ap.add_argument("--no-sky-gpu", dest="sky_gpu", action="store_false")
    ap.add_argument("--sky-refine", dest="sky_refine", action="store_true",
                    default=sky_cfg.get("refine", False))
    ap.add_argument("--no-sky-refine", dest="sky_refine", action="store_false")
    ap.add_argument("--sky-refine-radius", type=int,
                    default=sky_cfg.get("refine_radius", 24))
    ap.add_argument("--sky-refine-eps", type=float,
                    default=sky_cfg.get("refine_eps", 1e-3))
    ap.add_argument("--sky-refine-low", type=float,
                    default=sky_cfg.get("refine_low", 0.3))
    ap.add_argument("--sky-refine-high", type=float,
                    default=sky_cfg.get("refine_high", 0.5))
    ap.add_argument("--sky-refine-bias", type=float,
                    default=sky_cfg.get("refine_bias", 0.8))
    ap.add_argument("--sky-refine-no-bilateral", dest="sky_refine_no_bilateral",
                    action="store_true",
                    default=not sky_cfg.get("refine_bilateral", True))
    ap.add_argument("--sky-refine-bilateral", dest="sky_refine_no_bilateral",
                    action="store_false")

    # ---- quality control ----
    ap.add_argument("--qc", dest="qc_enabled", action="store_true",
                    default=qc_cfg.get("enabled", True))
    ap.add_argument("--no-qc", dest="qc_enabled", action="store_false")
    ap.add_argument("--qc-subdir", default=qc_cfg.get("subdir", "_review"))
    ap.add_argument("--qc-min-spread", type=float,
                    default=qc_cfg.get("min_spread", 1.5))
    ap.add_argument("--qc-min-s", type=float,
                    default=qc_cfg.get("min_scale", 0.005))
    ap.add_argument("--qc-min-inl", type=float,
                    default=qc_cfg.get("min_inlier_ratio", 0.3))

    # Legacy factory depth-to-color overrides used by color/infra1 modes.
    ap.add_argument("--d455-d2c-R", type=float, nargs=9, default=None,
                    help="override D455 Depth->Color rotation (row-major 3x3)")
    ap.add_argument("--d455-d2c-t", type=float, nargs=3, default=None,
                    help="override D455 Depth->Color translation")
    ap.add_argument("--d435-d2c-R", type=float, nargs=9, default=None,
                    help="override D435 Depth->Color rotation (row-major 3x3)")
    ap.add_argument("--d435-d2c-t", type=float, nargs=3, default=None,
                    help="override D435 Depth->Color translation")
    args = ap.parse_args()

    if not args.bag:
        ap.error("--bag is required unless io.bag is set in the GT config")

    # CLI paths are relative to the current directory; YAML paths were already
    # resolved relative to gt_config.yaml above.
    for attr in ("bag", "vis_dir", "export_dir", "deploy_mask", "left_calib",
                 "d435_calib", "sky_param", "sky_bin"):
        value = getattr(args, attr)
        if value:
            setattr(args, attr, str(Path(value).expanduser().resolve()))

    if gt_cfg_path is not None:
        print(f"GT config: {gt_cfg_path}")

    left_calibration = None
    if args.left_calib:
        left_calibration = load_left_d455(args.left_calib)
        T_LI = left_calibration.T_left_d455
        CAM1_PROJ = np.asarray(
            args.left_proj if args.left_proj is not None
            else left_calibration.left.intrinsics, dtype=float)
        CAM1_DIST = np.asarray(
            args.left_dist if args.left_dist is not None
            else left_calibration.left.distortion, dtype=float)
        CAM0_PROJ = np.asarray(left_calibration.d455.intrinsics, dtype=float)
        print(f"Kalibr left/D455: {left_calibration.source_path}")
        print(f"  {left_calibration.left.topic} <- {left_calibration.d455.topic}")
        print(f"  T_Left<-D455 t={np.round(T_LI[:3, 3], 4)}")

    d435_calibration = None
    if args.d435_calib:
        d435_calibration = load_d455_d435(args.d435_calib)
        print(f"Kalibr D455/D435: {d435_calibration.source_path}")
        print(f"  {d435_calibration.d455.topic} <- {d435_calibration.d435.topic}")
        print(f"  T_D455<-D435 t="
              f"{np.round(d435_calibration.T_d455_d435[:3, 3], 4)}")

    if left_calibration is not None and d435_calibration is not None:
        left_k = left_calibration.d455.intrinsics
        pair_k = d435_calibration.d455.intrinsics
        relative_difference = np.abs(left_k - pair_k) / np.maximum(np.abs(left_k), 1e-9)
        if np.any(relative_difference > 0.02):
            print("  WARNING: D455 intrinsics differ by >2% between the two "
                  "Kalibr files; the D455 CameraInfo/--d455-proj takes precedence.")

    device = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    os.makedirs(args.vis_dir, exist_ok=True)
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)

    review_vis = os.path.join(args.vis_dir, args.qc_subdir)
    review_export = (os.path.join(args.export_dir, args.qc_subdir)
                     if args.export_dir else None)
    if args.qc_enabled:
        os.makedirs(review_vis, exist_ok=True)
        if review_export:
            os.makedirs(review_export, exist_ok=True)
        print(f"  QC gate ON: suspicious outputs -> {args.qc_subdir}/")

    topics = [args.d455_depth, args.d455_info,
              args.left_topic, args.disp_topic]
    if args.d435_depth:
        topics.append(args.d435_depth)
    if args.d435_info:
        topics.append(args.d435_info)
    print("reading bag...")
    data = load_topics(args.bag, topics)
    for t in topics:
        print(f"  {t}: {len(data[t])}")

    for required_topic in (args.d455_depth, args.left_topic, args.disp_topic):
        if not data.get(required_topic):
            raise SystemExit(
                f"ERROR: required topic '{required_topic}' has 0 messages. "
                "Check the bag, GT config, or CLI topic override.")

    # Infra1 intrinsics for back-projection: config wins; otherwise CameraInfo,
    # then the legacy module default. This keeps each calibration set isolated.
    configured_d455_k = (args.d455_proj if args.d455_proj is not None
                         else gt_cfg.get("d455", {}).get("intrinsics"))
    if configured_d455_k is not None:
        K_I = tuple(np.asarray(configured_d455_k, dtype=float))
        print(f"  D455 K_I from config/CLI = {tuple(round(v,2) for v in K_I)}")
    elif data[args.d455_info]:
        k = np.array(data[args.d455_info][0][1].k).reshape(3, 3)
        K_I = (k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rel = np.abs(np.array(K_I) - CAM0_PROJ) / CAM0_PROJ
        print(f"  depth camera_info K_I = {tuple(round(v,2) for v in K_I)}")
        if np.any(rel > 0.02):
            print("  WARNING: depth camera_info differs from Kalibr cam0 by >2%.")
    else:
        K_I = tuple(CAM0_PROJ)
        print(f"  no depth/camera_info; using Kalibr cam0 {tuple(round(v,2) for v in K_I)}")
    print(f"  T_L<-I translation = {np.round(T_LI[:3,3],4)}")

    print("loading Depth-Anything-V2...")
    proc, model = load_da_v2(args.da_model, device)

    out_hw = image_to_numpy(data[args.left_topic][0][1]).shape[:2]
    print(f"  stereo-left size (HxW) = {out_hw}  device={device}")

    deploy_mask = None
    if args.deploy_mask:
        deploy_mask = load_deploy_mask(args.deploy_mask, out_hw)
        print(f"  deploy mask {args.deploy_mask}: coverage {100*deploy_mask.mean():.1f}%")
    else:
        print("  no --deploy-mask: using the whole target image as the FOV.")

    # sky segmentation
    sky_seg = None
    if not args.sky_enabled:
        print("  sky: DISABLED")
    elif args.sky_heuristic:
        print("  sky: brightness/texture heuristic")
    elif args.sky_param and args.sky_bin:
        sky_seg = SkySegmenter(
            args.sky_param, args.sky_bin, size=args.sky_size,
            input_name=args.sky_input_name, output_name=args.sky_output_name,
            mean=tuple(args.sky_mean), norm=tuple(args.sky_norm),
            sigmoid=not args.sky_no_sigmoid, thresh=args.sky_thresh,
            invert=args.sky_invert, use_gpu=args.sky_gpu,
            refine=args.sky_refine, refine_radius=args.sky_refine_radius,
            refine_eps=args.sky_refine_eps, refine_low=args.sky_refine_low,
            refine_high=args.sky_refine_high, refine_bias=args.sky_refine_bias,
            refine_bilateral=not args.sky_refine_no_bilateral)
        print(f"  sky: NCNN model {args.sky_param}"
              + (" + guided refinement" if args.sky_refine else ""))
    else:
        raise SystemExit("sky is enabled but --sky-param/--sky-bin are missing")

    # D435 (second RealSense) transform setup
    d435_enabled = bool(args.d435_depth) and len(data.get(args.d435_depth, [])) > 0
    if args.d435_depth and not d435_enabled:
        print(f"  D435: topic {args.d435_depth} has no messages; DISABLED")
    if d435_enabled:
        T_c455_i455 = (make_T(np.array(args.d455_d2c_R).reshape(3, 3), np.array(args.d455_d2c_t))
                       if args.d455_d2c_R is not None and args.d455_d2c_t is not None
                       else make_T(D455_D2C_R, D455_D2C_t))

        if args.d435_source == "kalibr_infra1":
            if d435_calibration is None:
                raise SystemExit(
                    "--d435-source kalibr_infra1 requires --d435-calib")
            if "infra1" not in d435_calibration.d435.topic.lower():
                raise SystemExit(
                    "kalibr_infra1 requires an infra1<->infra1 Kalibr file; "
                    f"got {d435_calibration.d435.topic}")
            T_L_D435 = T_LI @ d435_calibration.T_d455_d435
            d435_src_dist = d435_calibration.d435.distortion
            if args.d435_proj is not None:
                K_d435 = tuple(args.d435_proj)
            else:
                K_d435 = tuple(d435_calibration.d435.intrinsics)

        elif args.d435_source == "color":
            T_L_c455 = T_LI @ np.linalg.inv(T_c455_i455)
            if d435_calibration is not None:
                if "color" not in d435_calibration.d435.topic.lower():
                    raise SystemExit(
                        "color source requires a color<->color Kalibr file; "
                        f"got {d435_calibration.d435.topic}")
                T_L_D435 = T_L_c455 @ d435_calibration.T_d455_d435
                d435_src_dist = d435_calibration.d435.distortion
                K_d435 = (tuple(args.d435_proj) if args.d435_proj is not None
                          else tuple(d435_calibration.d435.intrinsics))
            else:
                T_L_D435 = compose_T_L_from_D435color(T_c455_i455)
                d435_src_dist = D435_COLOR_DIST
                K_d435 = (tuple(args.d435_proj) if args.d435_proj is not None
                          else tuple(D435_COLOR_PROJ))
        else:  # infra1 (raw depth) -- cleaner, matches the D455 path
            # Legacy color-chain path retained for reference.  It was needed
            # when only a D435-color <-> D455-color calibration was available:
            # T_c435_i435 = (make_T(np.array(args.d435_d2c_R).reshape(3, 3), np.array(args.d435_d2c_t))
            #                    if args.d435_d2c_R is not None and args.d435_d2c_t is not None
            #                    else make_T(D435_D2C_R, D435_D2C_t))
            # T_L_D435 = compose_T_L_from_D435infra1(T_c455_i455, T_c435_i435)

            # New direct raw-depth path using the 2026-08-27 Infra1 calibration:
            # D435 Infra1 -> D455 Infra1 -> stereo-left.
            T_L_D435 = T_LI @ np.linalg.inv(T_I435_I455)
            d435_src_dist = None                              # rectified infra1 = pinhole
            if args.d435_proj is not None:
                K_d435 = tuple(args.d435_proj)
            elif args.d435_info and data.get(args.d435_info):
                kk = np.array(data[args.d435_info][0][1].k).reshape(3, 3)
                K_d435 = (kk[0, 0], kk[1, 1], kk[0, 2], kk[1, 2])
            else:
                raise SystemExit("--d435-source infra1 needs --d435-proj or --d435-info "
                                 "(the D435 infra1 intrinsics differ from its color K)")
        print(f"  D435: enabled  source={args.d435_source}  topic={args.d435_depth}  "
              f"merge={args.d435_merge}")
        print(f"        K_d435={tuple(round(v,1) for v in K_d435)}  "
              f"T_L<-D435 t={np.round(T_L_D435[:3,3],4)}  "
              f"dist={'color radtan' if d435_src_dist is not None else 'none (pinhole)'}")

    n = 0
    n_suspect = 0
    n_skipped_sync = 0
    n_skipped_anchors = 0
    n_skipped_fit = 0
    for t_anchor, disp_msg in data[args.disp_topic]:
        if n >= args.max_pairs:
            break
        dm = nearest(data[args.d455_depth],
                     t_anchor - args.depth_time_offset, args.sync_tol)
        lm = nearest(data[args.left_topic], t_anchor, args.sync_tol)
        if dm is None or lm is None:
            if n_skipped_sync < 5:
                print(f"  [skip @ disp t={t_anchor:.3f}] "
                      f"depth={'MISS' if dm is None else 'ok'} "
                      f"left={'MISS' if lm is None else 'ok'}")
            n_skipped_sync += 1
            continue

        depth_I = image_to_numpy(dm[1]).astype(np.float32) * args.depth_scale   # infra1 frame
        left = image_to_numpy(lm[1])
        disp = image_to_numpy(disp_msg).astype(np.float32)
        if disp.ndim == 3:
            print("  disparity is 3-channel (colorized); log raw 32FC1. skipping.")
            continue

        disp_L = disp
        if disp.shape != out_hw:
            disp_L = cv2.resize(disp, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        disp_valid_L = np.isfinite(disp_L) & (disp_L > 0)

        # ---- STEP 1 ----
        da_L = step1_da_on_left(proc, model, left, device)
        # ---- STEP 2 (D455 infra1 depth -> left, single Kalibr hop) ----
        rs_depth_L = step2_ir_depth_to_L(depth_I, K_I, T_LI, CAM1_PROJ, CAM1_DIST,
                                         out_hw, splat=not args.no_splat)
        cover455 = float(np.isfinite(rs_depth_L).mean())

        # ---- STEP 2b (D435 depth -> left) merged in ----
        cover435 = 0.0
        if d435_enabled:
            d435m = nearest(data[args.d435_depth],
                            t_anchor - args.d435_time_offset, args.sync_tol)
            if d435m is not None:
                depth_435 = image_to_numpy(d435m[1]).astype(np.float32) * args.d435_depth_scale
                d435_L = step2_ir_depth_to_L(
                    depth_435, K_d435, T_L_D435, CAM1_PROJ, CAM1_DIST, out_hw,
                    splat=not args.no_splat, src_dist=d435_src_dist)
                cover435 = float(np.isfinite(d435_L).mean())
                rs_depth_L = merge_depth_L(rs_depth_L, d435_L, mode=args.d435_merge)

        fov = (deploy_mask if deploy_mask is not None
               else np.ones(out_hw, dtype=bool))

        # ---- SKY: segment, then EXCLUDE from the fit anchors ----
        left_bgr = left if left.ndim == 3 else cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if sky_seg is not None:
            sky_mask = sky_seg.mask(left_bgr) & fov
        elif args.sky_enabled and args.sky_heuristic:
            sky_mask = heuristic_sky_mask(left, fov)
        else:
            sky_mask = np.zeros(out_hw, dtype=bool)

        anchor_valid = (np.isfinite(rs_depth_L) & (rs_depth_L > args.dmin)
                        & (rs_depth_L < args.max_fit_depth) & fov & ~sky_mask)
        # anchor_valid = (np.isfinite(rs_depth_L) & (rs_depth_L > args.dmin)
        #                 & (rs_depth_L < args.max_fit_depth) & disp_valid_L & ~sky_mask)
        if anchor_valid.sum() < 50:
            if n_skipped_anchors < 5:
                print(f"  [skip @ disp t={t_anchor:.3f}] "
                      f"anchors={int(anchor_valid.sum())}<50  "
                      f"rs_cover={np.isfinite(rs_depth_L).mean()*100:.0f}% "
                      f"fov={fov.mean()*100:.0f}% sky={sky_mask.mean()*100:.0f}%")
            n_skipped_anchors += 1
            continue
        # ---- STEP 3 ----
        depth_L, info = step3_fit_metric_L(da_L, rs_depth_L, anchor_valid,
                                           args.da_metric, max_depth=args.gt_max_depth)
        if depth_L is None:
            if n_skipped_fit < 5:
                print(f"  [skip @ disp t={t_anchor:.3f}] fit returned None "
                      f"(anchors={int(anchor_valid.sum())})")
            n_skipped_fit += 1
            continue
        depth_L[sky_mask] = np.nan        # sky has no valid metric depth (viz + safety)
        # ---- STEP 4 ----
        # GT validity is independent of LightStereo disparity validity. The
        # disparity may contain model-failure holes that are not camera-FOV holes.
        # To restore the old behavior explicitly, append: & disp_valid_L
        gt_valid = fov & np.isfinite(depth_L) & (depth_L > 0) & ~sky_mask
        gt_final = np.where(gt_valid, depth_L, np.nan).astype(np.float32)

        suspect, reasons = (False, [])
        if args.qc_enabled:
            suspect, reasons = fit_is_suspect(
                depth_L, gt_valid, info, args.qc_min_spread,
                args.qc_min_s, args.qc_min_inl)
        vis_dir = review_vis if suspect else args.vis_dir
        export_dir = review_export if suspect else args.export_dir

        # ---- STEP 6 (viz) ----
        montage = build_montage(left, da_L, depth_I, rs_depth_L, anchor_valid,
                                depth_L, info, fov, gt_valid, disp_L,
                                args.tile, args.dmin, args.dmax, sky_mask=sky_mask)
        cv2.imwrite(os.path.join(vis_dir, f"gt_{n:04d}.png"), montage)

        # ---- STEP 5 (export) ----
        if export_dir:
            np.savez_compressed(
                os.path.join(export_dir, f"sample_{n:04d}.npz"),
                disp=disp.astype(np.float32),
                depth_aligned=gt_final,
                left=left,
                valid_mask=gt_valid.astype(np.bool_),
                sky_mask=(sky_mask & fov).astype(np.bool_),
                has_disp=np.bool_(True),
                stamp=np.float64(t_anchor),
            )

        if suspect:
            n_suspect += 1
            print(f"  [review] frame {n} -> {args.qc_subdir}/ "
                  f"({', '.join(reasons)})")

        if n % 20 == 0:
            fitmsg = (f"s={info['s']:.2f} inl={info['inl']*100:.0f}%"
                      if info.get("mode") == "affine" else f"x{info.get('ratio',1):.2f}")
            cov_txt = (f"cover D455={cover455*100:.0f}% D435={cover435*100:.0f}% "
                       f"merged={np.isfinite(rs_depth_L).mean()*100:.0f}%"
                       if d435_enabled else
                       f"RS->L cover={np.isfinite(rs_depth_L).mean()*100:.0f}%")
            print(f"  frame {n}: {cov_txt}  anchors={int(anchor_valid.sum())}  "
                  f"fit[{fitmsg}]  GT valid={gt_valid.mean()*100:.0f}%")
        n += 1

    print(f"done. {n} frames -> {args.vis_dir}"
          + (f" and {args.export_dir}" if args.export_dir else ""))
    total_disparity = len(data[args.disp_topic])
    checked = min(total_disparity, args.max_pairs)
    print(f"SKIP SUMMARY (of {checked} disparity frames):")
    print(f"  sync-miss   : {n_skipped_sync}")
    print(f"  <50 anchors : {n_skipped_anchors}")
    print(f"  fit failed  : {n_skipped_fit}")
    print(f"  kept        : {n}")
    if args.qc_enabled:
        print(f"QC: {n - n_suspect} kept, {n_suspect} moved to "
              f"{args.qc_subdir}/")


if __name__ == "__main__":
    main()
