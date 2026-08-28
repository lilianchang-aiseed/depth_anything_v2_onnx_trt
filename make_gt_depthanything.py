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
  4. Apply deploy FOV mask (masks/pair_1_0.npy).
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

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

# --------------------------------------------------------------------------- #
# Calibration (from calib_ir_...-camchain-0.yaml)
#   cam0 = infra1 (/camera/camera/infra1/image_rect_raw), 640x480 pinhole+radtan
#   cam1 = stereo-left (/stereo_1_0/left/image_rect),      320x320 pinhole+radtan
#   cam1.T_cn_cnm1 = T_{L<-I}  (maps a point in infra1 -> stereo-left)
# --------------------------------------------------------------------------- #
CAM0_PROJ = np.array([391.1833053665675, 391.2023258821923,
                      315.333377935706, 240.48835472327764])            # infra1 fx fy cx cy
CAM1_PROJ = np.array([167.07658738687104, 158.31284216746292,
                      138.2772130077489, 156.41669349148265])           # left  fx fy cx cy
CAM1_DIST = np.array([-0.032352588466769784, -0.0244944119435387,
                      -0.011755132756719569, -0.018714907886535945])    # left  k1 k2 p1 p2

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


# =========================================================================== #
# Sky segmentation (NCNN)  -- filters sky out of the affine fit and exports a
# sky mask so train_sml_global.py can label sky as a fixed far depth.
# =========================================================================== #
def _bias(x, b=0.8):
    """Bias curve from the paper (eq. 2), as in mask_refine.cpp::bias()."""
    return x / (((1.0 / b) - 2.0) * (1.0 - x) + 1.0)


def probability_to_confidence(prob, low=0.3, high=0.5, bias_b=0.8, eps=0.01):
    """
    Port of mask_refine.cpp::probability_to_confidence.

    Hysteresis on the raw sky probability -> a per-pixel CONFIDENCE weight:
      * prob < low   -> confident NON-sky, weight = bias((low-p)/low)
      * prob > high  -> confident sky,     weight = bias((p-high)/(1-high))
      * in between    -> eps (distrust the fuzzy boundary)
    """
    conf = np.full_like(prob, eps, dtype=np.float32)
    lo = prob < low
    hi = prob > high
    conf[lo] = np.maximum(_bias((low - prob[lo]) / low, bias_b), eps)
    conf[hi] = np.maximum(_bias((prob[hi] - high) / (1.0 - high), bias_b), eps)
    return conf


def refine_sky_prob(prob, gray, radius=24, eps=1e-3, low=0.3, high=0.5,
                    bias_b=0.8, do_bilateral=True):
    """
    Confidence-weighted guided-filter refinement of a sky probability map.

    This is the compact equivalent of mask_refine.cpp: that file fits a local
    affine RGB->mask model, weighted by the hysteresis confidence, solved at low
    resolution (the LDL solve) and upsampled -- which is exactly a *guided
    filter with a confidence weight*. Because the stereo-left image is grayscale
    (an RGB guide with 3 identical channels makes the 3x3 covariance singular),
    the scalar guided-filter form is used: stable and identical in effect.

    Effect: the mask boundary snaps to the actual image edge (the horizon /
    skyline), and the fuzzy CNN boundary is cleaned up. Larger `radius` => longer
    range snapping; `eps` controls edge sharpness.
    """
    p = prob.astype(np.float32)
    I = gray.astype(np.float32)
    if I.max() > 1.5:
        I = I / 255.0
    w = probability_to_confidence(p, low, high, bias_b)                 # confidence weight
    ksz = (2 * radius + 1, 2 * radius + 1)
    box = lambda a: cv2.boxFilter(a, -1, ksz, normalize=False,
                                  borderType=cv2.BORDER_REPLICATE)
    W = box(w) + 1e-8
    mean_I  = box(w * I) / W
    mean_p  = box(w * p) / W
    mean_II = box(w * I * I) / W
    mean_Ip = box(w * I * p) / W
    var_I  = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)                                          # local slope
    b = mean_p - a * mean_I                                             # local offset
    q = box(w * a) / W * I + box(w * b) / W                            # apply affine
    q = np.clip(q, 0.0, 1.0)
    if do_bilateral:                                                    # final smoothing (C++ does too)
        q = cv2.bilateralFilter(q.astype(np.float32), 0, 0.08, 8)
    return q


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
        p = self.prob(bgr)
        if self.refine:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            p = refine_sky_prob(p, gray, radius=self.refine_radius,
                                eps=self.refine_eps, low=self.refine_low,
                                high=self.refine_high, bias_b=self.refine_bias,
                                do_bilateral=self.refine_bilateral)
        return p > self.thresh


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def fit_is_suspect(depth_L, gt_valid, info, min_spread, min_s, min_inl):
    """
    Flag a collapsed / poorly-fit frame: one where almost all pixels map to a
    similar depth (affine slope ~0), or the fit had few inliers.

    Returns (suspect: bool, reasons: list[str]).
    """
    reasons = []
    d = depth_L[gt_valid & np.isfinite(depth_L)]
    if d.size < 50:
        return True, ["too_few_valid"]
    p10, p90 = np.percentile(d, [10, 90])
    spread = p90 / max(p10, 1e-6)                 # ~1.0 => all pixels same depth
    if spread < min_spread:
        reasons.append(f"spread={spread:.2f}<{min_spread}")
    if info.get("mode") == "affine":
        if abs(info.get("s", 0.0)) < min_s:
            reasons.append(f"s={info.get('s',0):.4f}<{min_s}")
        if info.get("inl", 1.0) < min_inl:
            reasons.append(f"inl={info.get('inl',0):.2f}<{min_inl}")
    return (len(reasons) > 0), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--vis-dir", default="./gt_vis")
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--deploy-mask", default=None,
                    help="masks/pair_1_0.npy; else disparity>0 is the FOV mask.")
    ap.add_argument("--da-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--da-metric", action="store_true")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--sync-tol", type=float, default=0.03)
    ap.add_argument("--max-pairs", type=int, default=1000)
    ap.add_argument("--dmin", type=float, default=0.2)
    ap.add_argument("--dmax", type=float, default=30.0)
    ap.add_argument("--max-fit-depth", type=float, default=30.0)
    ap.add_argument("--no-splat", action="store_true",
                    help="disable 2x2 splat (leave sub-pixel holes in the warp)")
    ap.add_argument("--tile", type=int, default=300)
    # ---- sky segmentation (NCNN) ----
    ap.add_argument("--sky-param", default=None, help="NCNN .param for sky seg")
    ap.add_argument("--sky-bin", default=None, help="NCNN .bin for sky seg")
    ap.add_argument("--sky-size", type=int, default=320)
    ap.add_argument("--sky-input-name", default="in0")
    ap.add_argument("--sky-output-name", default="out0")
    ap.add_argument("--sky-mean", type=float, nargs=3, default=[123.675, 116.28, 103.53])
    ap.add_argument("--sky-norm", type=float, nargs=3, default=[0.01712, 0.01751, 0.01743])
    ap.add_argument("--sky-no-sigmoid", action="store_true",
                    help="model already outputs probabilities (skip sigmoid)")
    ap.add_argument("--sky-thresh", type=float, default=0.5)
    ap.add_argument("--sky-invert", action="store_true",
                    help="flip if the model labels sky=0 / scene=1")
    ap.add_argument("--sky-heuristic", action="store_true",
                    help="use the brightness/texture heuristic instead of NCNN")
    ap.add_argument("--sky-gpu", action="store_true")
    # ---- sky mask refinement (guided-filter port of mask_refine.cpp) ----
    ap.add_argument("--sky-refine", action="store_true",
                    help="snap the sky mask to image edges via confidence-weighted "
                         "guided filtering (port of mask_refine.cpp)")
    ap.add_argument("--sky-refine-radius", type=int, default=24,
                    help="guided-filter radius; larger = longer-range edge snapping")
    ap.add_argument("--sky-refine-eps", type=float, default=1e-3,
                    help="guided-filter regularization; smaller = sharper edges")
    ap.add_argument("--sky-refine-low", type=float, default=0.3,
                    help="hysteresis low threshold (confident non-sky below this)")
    ap.add_argument("--sky-refine-high", type=float, default=0.5,
                    help="hysteresis high threshold (confident sky above this)")
    ap.add_argument("--sky-refine-bias", type=float, default=0.8)
    ap.add_argument("--sky-refine-no-bilateral", action="store_true",
                    help="skip the final bilateral filter")
    ap.add_argument("--depth-time-offset", type=float, default=0.0,
                    help="seconds subtracted from the disp time when matching D455 depth")
    ap.add_argument("--gt-max-depth", type=float, default=20.0,
                    help="mark GT pixels beyond this range NaN instead of clipping")
    # ---- second RealSense (D435), mounted below the D455 ----
    ap.add_argument("--d435-depth", default=None,
                    help="D435 depth topic; enables the 2nd warp. Pair with "
                         "--d435-source to say which frame it's in.")
    ap.add_argument("--d435-source", choices=["color", "infra1"], default="color",
                    help="color: topic is aligned_depth_to_color (D435 color frame); "
                         "infra1: topic is raw depth/image_rect_raw (D435 infra1 frame)")
    ap.add_argument("--d435-proj", type=float, nargs=4, default=None,
                    help="D435 intrinsics fx fy cx cy for the chosen source "
                         "(default: color=Kalibr color K; infra1=read from its camera_info)")
    ap.add_argument("--d435-info", default=None,
                    help="D435 camera_info topic to read intrinsics for --d435-source infra1")
    ap.add_argument("--d435-depth-scale", type=float, default=0.001)
    ap.add_argument("--d435-time-offset", type=float, default=0.0)
    ap.add_argument("--d435-merge", choices=["fill", "min"], default="fill",
                    help="fill: D455 wins, D435 fills holes; min: nearer of the two")
    ap.add_argument("--d455-d2c-R", type=float, nargs=9, default=None,
                    help="override D455 Depth->Color rotation (row-major 3x3)")
    ap.add_argument("--d455-d2c-t", type=float, nargs=3, default=None,
                    help="override D455 Depth->Color translation")
    ap.add_argument("--d435-d2c-R", type=float, nargs=9, default=None,
                    help="override D435 Depth->Color rotation (row-major 3x3)")
    ap.add_argument("--d435-d2c-t", type=float, nargs=3, default=None,
                    help="override D435 Depth->Color translation")
    # ---- inline QC: route collapsed/poor fits to a review subfolder ----
    ap.add_argument("--no-qc", action="store_true",
                    help="disable the inline collapsed-fit gate")
    ap.add_argument("--qc-subdir", default="_review",
                    help="subfolder under vis/export for flagged samples")
    ap.add_argument("--qc-min-spread", type=float, default=1.5,
                    help="flag if p90/p10 GT-depth ratio below this (near-constant depth)")
    ap.add_argument("--qc-min-s", type=float, default=0.005,
                    help="flag if |affine slope| below this (fit collapsed to s~0)")
    ap.add_argument("--qc-min-inl", type=float, default=0.3,
                    help="flag if fit inlier fraction below this")
    args = ap.parse_args()

    device = "cuda"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    os.makedirs(args.vis_dir, exist_ok=True)
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)

    qc_on = not args.no_qc
    review_vis = os.path.join(args.vis_dir, args.qc_subdir)
    review_export = os.path.join(args.export_dir, args.qc_subdir) if args.export_dir else None
    if qc_on:
        os.makedirs(review_vis, exist_ok=True)
        if review_export:
            os.makedirs(review_export, exist_ok=True)
        print(f"  QC gate ON: collapsed fits -> {args.qc_subdir}/ "
              f"(min_spread={args.qc_min_spread} min_s={args.qc_min_s} "
              f"min_inl={args.qc_min_inl})")

    topics = [TOPIC_DEPTH, TOPIC_DEPTH_INFO, TOPIC_LEFT, TOPIC_DISP]
    if args.d435_depth:
        topics.append(args.d435_depth)
    if args.d435_info:
        topics.append(args.d435_info)
    print("reading bag...")
    data = load_topics(args.bag, topics)
    for t in topics:
        print(f"  {t}: {len(data[t])}")

    # infra1 intrinsics for back-projection: prefer depth/camera_info, else Kalibr cam0
    if data[TOPIC_DEPTH_INFO]:
        k = np.array(data[TOPIC_DEPTH_INFO][0][1].k).reshape(3, 3)
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

    out_hw = image_to_numpy(data[TOPIC_LEFT][0][1]).shape[:2]
    print(f"  stereo-left size (HxW) = {out_hw}  device={device}")

    deploy_mask = None
    if args.deploy_mask:
        deploy_mask = load_deploy_mask(args.deploy_mask, out_hw)
        print(f"  deploy mask {args.deploy_mask}: coverage {100*deploy_mask.mean():.1f}%")
    else:
        print("  no --deploy-mask: using recorded disparity>0 as the FOV mask.")

    # sky segmentation
    sky_seg = None
    if args.sky_heuristic:
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
              + ("  + refine (guided filter)" if args.sky_refine else ""))
    else:
        print("  sky: DISABLED (no --sky-param/--sky-bin, no --sky-heuristic)")

    # D435 (second RealSense) transform setup
    d435_enabled = bool(args.d435_depth) and len(data.get(args.d435_depth, [])) > 0
    if args.d435_depth and not d435_enabled:
        print(f"  D435: topic {args.d435_depth} has no messages; DISABLED")
    if d435_enabled:
        T_c455_i455 = (make_T(np.array(args.d455_d2c_R).reshape(3, 3), np.array(args.d455_d2c_t))
                       if args.d455_d2c_R is not None and args.d455_d2c_t is not None
                       else make_T(D455_D2C_R, D455_D2C_t))

        if args.d435_source == "color":
            T_L_D435 = compose_T_L_from_D435color(T_c455_i455)
            d435_src_dist = D435_COLOR_DIST                    # color lens distortion
            if args.d435_proj is not None:
                K_d435 = tuple(args.d435_proj)
            else:
                K_d435 = tuple(D435_COLOR_PROJ)
        else:  # infra1 (raw depth) -- cleaner, matches the D455 path
            T_c435_i435 = (make_T(np.array(args.d435_d2c_R).reshape(3, 3), np.array(args.d435_d2c_t))
                           if args.d435_d2c_R is not None and args.d435_d2c_t is not None
                           else make_T(D435_D2C_R, D435_D2C_t))
            T_L_D435 = compose_T_L_from_D435infra1(T_c455_i455, T_c435_i435)
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
    for t_anchor, disp_msg in data[TOPIC_DISP]:
        if n >= args.max_pairs:
            break
        dm = nearest(data[TOPIC_DEPTH], t_anchor - args.depth_time_offset, args.sync_tol)
        lm = nearest(data[TOPIC_LEFT], t_anchor, args.sync_tol)
        if dm is None or lm is None:
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

        fov = deploy_mask if deploy_mask is not None else disp_valid_L

        # SKY region-of-interest: the fisheye DISK (image content), NOT `fov`.
        # `fov` for anchors is disp_valid_L when no --deploy-mask is given, but
        # sky has near-zero disparity -> `& fov` here would erase ~all sky.
        # Use the deploy mask if we have one (it IS the disk); otherwise infer
        # the disk from left-image brightness (>8 = has image content).
        if deploy_mask is not None:
            sky_fov = deploy_mask
        else:
            g8 = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            sky_fov = g8 > 8

        # ---- SKY: segment, then EXCLUDE from the fit anchors ----
        left_bgr = left if left.ndim == 3 else cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if sky_seg is not None:
            sky_mask = sky_seg.mask(left_bgr) & sky_fov
        elif args.sky_heuristic:
            sky_mask = heuristic_sky_mask(left, sky_fov)
        else:
            sky_mask = np.zeros(out_hw, dtype=bool)

        anchor_valid = (np.isfinite(rs_depth_L) & (rs_depth_L > args.dmin)
                        & (rs_depth_L < args.max_fit_depth) & fov & ~sky_mask)
        if anchor_valid.sum() < 50:
            continue
        # ---- STEP 3 ----
        depth_L, info = step3_fit_metric_L(da_L, rs_depth_L, anchor_valid,
                                           args.da_metric, max_depth=args.gt_max_depth)
        if depth_L is None:
            continue
        depth_L[sky_mask] = np.nan        # sky has no valid metric depth (viz + safety)
        # ---- STEP 4 ----
        # GT valid where the fit is reliable AND disparity exists AND NOT sky.
        # Sky is exported as its own mask; train_sml_global.py assigns it a fixed
        # far depth so it doesn't need a metric value here.
        gt_valid = fov & np.isfinite(depth_L) & (depth_L > 0) & disp_valid_L & ~sky_mask
        gt_final = np.where(gt_valid, depth_L, np.nan).astype(np.float32)

        # ---- QC: is this a collapsed / poor fit? ----
        suspect, reasons = (False, [])
        if qc_on:
            suspect, reasons = fit_is_suspect(
                depth_L, gt_valid, info,
                args.qc_min_spread, args.qc_min_s, args.qc_min_inl)
        vis_dir = review_vis if (qc_on and suspect) else args.vis_dir
        exp_dir = (review_export if (qc_on and suspect) else args.export_dir)

        # ---- STEP 6 (viz) ----
        montage = build_montage(left, da_L, depth_I, rs_depth_L, anchor_valid,
                                depth_L, info, fov, gt_valid, disp_L,
                                args.tile, args.dmin, args.dmax, sky_mask=sky_mask)
        cv2.imwrite(os.path.join(vis_dir, f"gt_{n:04d}.png"), montage)

        # ---- STEP 5 (export) ----
        if exp_dir:
            np.savez_compressed(
                os.path.join(exp_dir, f"sample_{n:04d}.npz"),
                disp=disp.astype(np.float32),
                depth_aligned=gt_final,
                left=left,
                valid_mask=gt_valid.astype(np.bool_),
                sky_mask=(sky_mask & sky_fov).astype(np.bool_),
                has_disp=np.bool_(True),
                stamp=np.float64(t_anchor),
            )

        if suspect:
            n_suspect += 1
            print(f"  [review] frame {n} -> {args.qc_subdir}/  ({', '.join(reasons)})")

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
    if qc_on:
        kept = n - n_suspect
        print(f"QC: {kept} kept, {n_suspect} flagged to {args.qc_subdir}/ "
              f"({100*n_suspect/max(n,1):.0f}%). Review the montages there and move "
              f"any good ones back up a level.")


if __name__ == "__main__":
    main()