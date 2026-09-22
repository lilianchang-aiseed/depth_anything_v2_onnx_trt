#!/usr/bin/env python3
"""
Clamped metric GT for DA-V2 training -- DA-V2 on the LEFT rect image, anchored by the
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
  3. Near-weighted MSE affine fit DA(left) -> dense metric depth.
  4. Keep exact metric targets through 20 m; mark far as 19 m and sky as 20 m.
  5. Export separate metric_valid_mask, far_mask, and sky_mask supervision masks.
  6. Export a 3x4 fitting, mask, depth, overlay, and residual montage.

Deps: rosbags, numpy, opencv-python, torch, transformers, pillow
Output NPZ content in the bottom of the code
"""
import argparse
import csv
import os
import re
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from gt_fitting import select_fit_validation_masks, robust_affine_invdepth, _pava_increasing, _pchip_slopes, _pchip_evaluate, isotonic_pchip_invdepth, validation_depth_statistics, step3_fit_metric_L

ML_GT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_GT_ROOT))
from tools.correct_matrix_direction import load_d455_d435, load_left_d455
try:
    from .ege_ncnn_sky import EgeNcnnSkySegmenter
except ImportError:  # direct execution
    from ege_ncnn_sky import EgeNcnnSkySegmenter

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
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

_RUNTIME_DEFAULTS = {
    "CAM0_PROJ": CAM0_PROJ.copy(),
    "CAM1_PROJ": CAM1_PROJ.copy(),
    "CAM1_DIST": CAM1_DIST.copy(),
    "T_LI": T_LI.copy(),
    "T_I435_I455": T_I435_I455.copy(),
    "TOPIC_DEPTH": TOPIC_DEPTH,
    "TOPIC_DEPTH_INFO": TOPIC_DEPTH_INFO,
    "TOPIC_LEFT": TOPIC_LEFT,
    "TOPIC_DISP": TOPIC_DISP,
}


def reset_runtime_defaults():
    """Prevent one multibag config leaking calibration into the next."""
    global CAM0_PROJ, CAM1_PROJ, CAM1_DIST, T_LI, T_I435_I455
    global TOPIC_DEPTH, TOPIC_DEPTH_INFO, TOPIC_LEFT, TOPIC_DISP
    CAM0_PROJ = _RUNTIME_DEFAULTS["CAM0_PROJ"].copy()
    CAM1_PROJ = _RUNTIME_DEFAULTS["CAM1_PROJ"].copy()
    CAM1_DIST = _RUNTIME_DEFAULTS["CAM1_DIST"].copy()
    T_LI = _RUNTIME_DEFAULTS["T_LI"].copy()
    T_I435_I455 = _RUNTIME_DEFAULTS["T_I435_I455"].copy()
    TOPIC_DEPTH = _RUNTIME_DEFAULTS["TOPIC_DEPTH"]
    TOPIC_DEPTH_INFO = _RUNTIME_DEFAULTS["TOPIC_DEPTH_INFO"]
    TOPIC_LEFT = _RUNTIME_DEFAULTS["TOPIC_LEFT"]
    TOPIC_DISP = _RUNTIME_DEFAULTS["TOPIC_DISP"]


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


def load_topics_sampled(bagpath, topics, disp_topic, frame_interval, sync_tol,
                        topic_time_offsets, info_topics):
    """Single-pass bag scan retaining only messages near sampled frames."""
    out = {topic: [] for topic in topics}
    scanned = {topic: 0 for topic in topics}
    sampled_topics = {
        topic for topic in topics
        if topic != disp_topic and topic not in info_topics
    }
    recent = {topic: deque() for topic in sampled_topics}
    active_targets = {topic: deque() for topic in sampled_topics}
    history_window = max(
        1.0,
        4.0 * sync_tol +
        max((abs(value) for value in topic_time_offsets.values()), default=0.0),
    )

    with AnyReader([Path(bagpath)], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=conns):
            topic = conn.topic
            source_frame = scanned[topic]
            scanned[topic] += 1
            if topic == disp_topic and source_frame % frame_interval != 0:
                continue
            if topic in info_topics and out[topic]:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = stamp_to_sec(msg)
            if topic in info_topics:
                out[topic].append((stamp, msg))
                continue
            if topic == disp_topic:
                out[topic].append((stamp, msg))
                for sampled_topic in sampled_topics:
                    target = stamp + topic_time_offsets.get(sampled_topic, 0.0)
                    for recent_stamp, recent_msg in recent[sampled_topic]:
                        if abs(recent_stamp - target) <= sync_tol:
                            out[sampled_topic].append((recent_stamp, recent_msg))
                    active_targets[sampled_topic].append(target)
                continue

            topic_recent = recent[topic]
            topic_recent.append((stamp, msg))
            while topic_recent and topic_recent[0][0] < stamp - history_window:
                topic_recent.popleft()

            targets = active_targets[topic]
            while targets and stamp > targets[0] + sync_tol:
                targets.popleft()
            if any(abs(stamp - target) <= sync_tol for target in targets):
                out[topic].append((stamp, msg))

    for topic in out:
        out[topic].sort(key=lambda item: item[0])
    return out, scanned


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


class GTProcessor:
    """Reusable model cache for sequential single- or multi-bag processing."""

    def __init__(self):
        self._da_models = {}
        self._sky_models = {}

    def load_da_v2(self, model_id, device):
        key = (str(model_id), str(device))
        if key not in self._da_models:
            self._da_models[key] = load_da_v2(model_id, device)
        else:
            print(f"reusing cached Depth-Anything-V2: {model_id}")
        return self._da_models[key]

    def load_sky(self, args):
        key = (
            str(Path(args.sky_param).resolve()),
            str(Path(args.sky_bin).resolve()),
            int(args.sky_size), args.sky_input_name, args.sky_output_name,
            tuple(args.sky_mean), tuple(args.sky_norm), float(args.sky_thresh),
            bool(args.sky_invert), bool(args.sky_gpu),
            bool(args.sky_dynamic_input_scale), bool(args.sky_refine),
            int(args.sky_refine_radius), float(args.sky_refine_eps),
            float(args.sky_refine_low), float(args.sky_refine_high),
            float(args.sky_refine_bias), bool(args.sky_refine_no_bilateral),
        )
        if key not in self._sky_models:
            self._sky_models[key] = EgeNcnnSkySegmenter(
                args.sky_param, args.sky_bin, size=args.sky_size,
                input_name=args.sky_input_name,
                output_name=args.sky_output_name,
                mean=tuple(args.sky_mean), norm=tuple(args.sky_norm),
                threshold=args.sky_thresh, invert=args.sky_invert,
                use_gpu=args.sky_gpu,
                dynamic_input_scale=args.sky_dynamic_input_scale,
                refine=args.sky_refine,
                refine_radius=args.sky_refine_radius,
                refine_eps=args.sky_refine_eps,
                refine_low=args.sky_refine_low,
                refine_high=args.sky_refine_high,
                refine_bias=args.sky_refine_bias,
                refine_bilateral=not args.sky_refine_no_bilateral,
            )
        else:
            print(f"reusing cached EGE sky model: {args.sky_param}")
        return self._sky_models[key]

    def process_bag(self, argv):
        return main(argv=argv, processor=self)


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


def deploy_mask_from_left_topic(left_topic):
    """Resolve the matching circle/wing mask for old and raw2rect topics."""
    masks_dir = Path(__file__).resolve().parent.parent / "masks"

    stereo_match = re.fullmatch(
        r"/stereo_(\d+)_(\d+)/left/image_rect", left_topic)
    if stereo_match is not None:
        pair = f"{stereo_match.group(1)}_{stereo_match.group(2)}"
        mask_path = masks_dir / f"pair_{pair}_circle_fov.npy"
    else:
        camera_match = re.fullmatch(r"/camera_(\d+)/image_rect", left_topic)
        raw2rect_pairs = {
            "0": "0_3",
            "1": "1_0",
            "2": "2_1",
            "3": "3_2",
        }
        if camera_match is None or camera_match.group(1) not in raw2rect_pairs:
            raise SystemExit(
                "cannot auto-select FOV mask: --left-topic must match "
                "'/stereo_<left>_<right>/left/image_rect' or "
                "'/camera_<0..3>/image_rect'; "
                f"got {left_topic!r}"
            )
        pair = raw2rect_pairs[camera_match.group(1)]
        # raw2rect currently emits 322x322 images, so use the masks generated
        # in that native resolution instead of resizing the older 320 masks.
        mask_path = masks_dir / "322x322" / f"pair_{pair}_circle_fov.npy"
    if not mask_path.is_file():
        raise SystemExit(
            f"auto-selected FOV mask does not exist: {mask_path}"
        )
    return str(mask_path)


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
# def _weighted_choice(rng, depths, count, near_sample_weight):
#     """Sample without replacement, giving anchors at <=5 m extra weight."""
#     count = min(int(count), len(depths))
#     if count >= len(depths):
#         return np.arange(len(depths))
#     weights = np.ones(len(depths), dtype=np.float64)
#     weights[np.asarray(depths) <= 5.0] = float(near_sample_weight)
#     return rng.choice(len(depths), count, replace=False,
#                       p=weights / weights.sum())


# def select_fit_validation_masks(valid, depths, fit_samples,
#                                 validation_ratio, near_sample_weight, rng=None):
#     """Create disjoint, near-weighted fitting and held-out validation masks."""
#     rng = rng if rng is not None else np.random.default_rng(0)
#     ys, xs = np.where(valid)
#     count = len(xs)
#     if count < 50:
#         return None, None

#     validation_count = int(round(count * validation_ratio))
#     validation_count = min(validation_count, max(0, count - 50))
#     validation_mask = np.zeros_like(valid, dtype=bool)
#     remaining = np.ones(count, dtype=bool)
#     if validation_count:
#         selected = _weighted_choice(
#             rng, depths[ys, xs], validation_count, near_sample_weight)
#         validation_mask[ys[selected], xs[selected]] = True
#         remaining[selected] = False

#     fit_ys, fit_xs = ys[remaining], xs[remaining]
#     selected = _weighted_choice(
#         rng, depths[fit_ys, fit_xs], fit_samples, near_sample_weight)
#     fit_mask = np.zeros_like(valid, dtype=bool)
#     fit_mask[fit_ys[selected], fit_xs[selected]] = True
#     return fit_mask, validation_mask


# def robust_affine_invdepth(rel, inv_gt, valid, ransac_iters=200,
#                            iters=3, k=2.5, rng=None):
#     rng = rng if rng is not None else np.random.default_rng(0)
#     ys, xs = np.where(valid)
#     if len(xs) < 50:
#         return None
#     x = rel[ys, xs].astype(np.float64)
#     y = inv_gt[ys, xs].astype(np.float64)
#     tau = 0.3 * (1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9)
#     bs, bt, bi, N = 1.0, 0.0, -1, len(x)
#     for _ in range(ransac_iters):
#         i, j = rng.integers(0, N, size=2)
#         if abs(x[i] - x[j]) < 1e-9:
#             continue
#         s = (y[i] - y[j]) / (x[i] - x[j]); t = y[i] - s * x[i]
#         inl = int((np.abs(y - (s * x + t)) < tau).sum())
#         if inl > bi:
#             bs, bt, bi = s, t, inl
#     s, t = bs, bt
#     keep = np.abs(y - (s * x + t)) < tau
#     for _ in range(iters):
#         if keep.sum() < 50:
#             break
#         A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
#         sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
#         s, t = float(sol[0]), float(sol[1])
#         res = y - (s * x + t); med = np.median(res[keep])
#         mad = 1.4826 * np.median(np.abs(res[keep] - med)) + 1e-9
#         keep = np.abs(res - med) < k * mad
#     return s, t, float(keep.mean()), int(keep.sum()), int(N)


# def _pava_increasing(values, weights):
#     """Weighted pool-adjacent-violators algorithm for nondecreasing values."""
#     levels, masses, starts, ends = [], [], [], []
#     for index, (value, weight) in enumerate(zip(values, weights)):
#         levels.append(float(value)); masses.append(float(weight))
#         starts.append(index); ends.append(index + 1)
#         while len(levels) >= 2 and levels[-2] > levels[-1]:
#             mass = masses[-2] + masses[-1]
#             level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / mass
#             levels[-2:] = [level]; masses[-2:] = [mass]
#             ends[-2:] = [ends[-1]]; starts.pop()
#     output = np.empty(len(values), dtype=np.float64)
#     for level, start, end in zip(levels, starts, ends):
#         output[start:end] = level
#     return output


# def _pchip_slopes(x, y):
#     """Fritsch-Carlson derivatives used by monotone cubic Hermite interpolation."""
#     if len(x) == 2:
#         slope = (y[1] - y[0]) / (x[1] - x[0])
#         return np.array([slope, slope], dtype=np.float64)
#     h = np.diff(x)
#     delta = np.diff(y) / h
#     slopes = np.zeros_like(y)
#     same = delta[:-1] * delta[1:] > 0
#     w1 = 2.0 * h[1:] + h[:-1]
#     w2 = h[1:] + 2.0 * h[:-1]
#     interior = np.flatnonzero(same) + 1
#     slopes[interior] = ((w1[same] + w2[same]) /
#                         (w1[same] / delta[:-1][same] +
#                          w2[same] / delta[1:][same]))

#     def endpoint(h0, h1, d0, d1):
#         value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
#         if np.sign(value) != np.sign(d0):
#             return 0.0
#         if np.sign(d0) != np.sign(d1) and abs(value) > abs(3.0 * d0):
#             return 3.0 * d0
#         return value

#     slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
#     slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
#     return slopes


# def _pchip_evaluate(x, y, query):
#     """Evaluate monotone PCHIP, clamping instead of extrapolating past anchors."""
#     query = np.asarray(query, dtype=np.float64)
#     clipped = np.clip(query, x[0], x[-1])
#     index = np.clip(np.searchsorted(x, clipped, side="right") - 1, 0, len(x) - 2)
#     h = x[index + 1] - x[index]
#     u = (clipped - x[index]) / h
#     slopes = _pchip_slopes(x, y)
#     return ((2*u**3 - 3*u**2 + 1) * y[index]
#             + (u**3 - 2*u**2 + u) * h * slopes[index]
#             + (-2*u**3 + 3*u**2) * y[index + 1]
#             + (u**3 - u**2) * h * slopes[index + 1])


# def isotonic_pchip_invdepth(rel, inv_gt, fit_mask, metric_depth_max):
#     """RANSAC-filtered, monotone DA-relative -> inverse-metric-depth curve."""
#     affine = robust_affine_invdepth(rel, inv_gt, fit_mask)
#     if affine is None:
#         return None
#     s, t, _, _, sample_count = affine
#     ys, xs = np.where(fit_mask)
#     x = rel[ys, xs].astype(np.float64)
#     y = inv_gt[ys, xs].astype(np.float64)
#     residual = y - (s * x + t)
#     median = np.median(residual)
#     mad = 1.4826 * np.median(np.abs(residual - median)) + 1e-9
#     keep = np.abs(residual - median) < 2.5 * mad
#     x, y = x[keep], y[keep]
#     if len(x) < 50:
#         return None

#     order = np.argsort(x)
#     x, y = x[order], y[order]
#     bins = np.array_split(np.arange(len(x)), min(128, len(x)))
#     knot_x = np.array([np.median(x[index]) for index in bins])
#     knot_y = np.array([np.median(y[index]) for index in bins])
#     knot_w = np.array([len(index) for index in bins], dtype=np.float64)
#     unique_x, inverse = np.unique(knot_x, return_inverse=True)
#     if len(unique_x) < 2:
#         return None
#     merged_y = np.zeros(len(unique_x), dtype=np.float64)
#     merged_w = np.zeros(len(unique_x), dtype=np.float64)
#     np.add.at(merged_y, inverse, knot_y * knot_w)
#     np.add.at(merged_w, inverse, knot_w)
#     merged_y /= merged_w
#     monotone_y = _pava_increasing(merged_y, merged_w)
#     mapped_inv = _pchip_evaluate(unique_x, monotone_y, rel)

#     threshold_inv = 1.0 / float(metric_depth_max)
#     unique_y, first = np.unique(monotone_y, return_index=True)
#     if threshold_inv <= unique_y[0]:
#         da_far_threshold = unique_x[0]
#     elif threshold_inv >= unique_y[-1]:
#         da_far_threshold = unique_x[-1]
#     else:
#         da_far_threshold = float(np.interp(threshold_inv, unique_y, unique_x[first]))
#     return mapped_inv, {
#         "mode": "isotonic-pchip", "s": s, "t": t,
#         "inl": float(keep.mean()), "inlier_count": int(keep.sum()),
#         "fit_sample_count": int(sample_count), "pchip_knots": int(len(unique_x)),
#         "da_min_anchor": float(unique_x[0]), "da_max_anchor": float(unique_x[-1]),
#         "da_far_threshold": float(da_far_threshold),
#     }


# def validation_depth_statistics(pred_depth, target_depth, validation_mask):
#     """Metric errors on anchors held out from scale/shift fitting."""
#     valid = (validation_mask & np.isfinite(pred_depth)
#              & np.isfinite(target_depth) & (target_depth > 0))
#     result = {
#         "validation_count": int(validation_mask.sum()),
#         "validation_valid_count": int(valid.sum()),
#         "val_mae_m": np.nan,
#         "val_rmse_m": np.nan,
#         "val_median_relative_error": np.nan,
#         "val_mae_lt5m": np.nan,
#         "val_mae_0_2m": np.nan,
#         "val_mae_2_5m": np.nan,
#         "val_mae_5_10m": np.nan,
#         "val_mae_10_20m": np.nan,
#     }
#     if not valid.any():
#         return result

#     target = target_depth[valid].astype(np.float64)
#     error = np.abs(pred_depth[valid].astype(np.float64) - target)
#     result.update({
#         "val_mae_m": float(error.mean()),
#         "val_rmse_m": float(np.sqrt(np.mean(error ** 2))),
#         "val_median_relative_error": float(np.median(error / target)),
#     })
#     for name, low, high in (
#         ("val_mae_0_2m", 0.0, 2.0),
#         ("val_mae_2_5m", 2.0, 5.0),
#         ("val_mae_5_10m", 5.0, 10.0),
#         ("val_mae_10_20m", 10.0, 20.0),
#     ):
#         selected = (target >= low) & (target < high)
#         if selected.any():
#             result[name] = float(error[selected].mean())
#     selected = target < 5.0
#     if selected.any():
#         result["val_mae_lt5m"] = float(error[selected].mean())
#     return result


# def step3_fit_metric_L(da_L, rs_depth_L, anchor_valid, da_metric=False,
#                        fit_mode="isotonic-pchip", metric_depth_max=15.0,
#                        fit_samples=10000,
#                        near_sample_weight=3.0, validation_ratio=0.2):
#     """
#     Fit DA(left) -> metric using the RS anchors, then convert the WHOLE da_L to
#     metric depth. The returned map is deliberately not upper-clipped: the caller
#     needs it to distinguish exact metric targets from the far class.
#     """
#     valid_anchor = (anchor_valid & np.isfinite(da_L) & (da_L > 0)
#                     & np.isfinite(rs_depth_L) & (rs_depth_L > 0))
#     rng = np.random.default_rng(0)
#     fit_mask, validation_mask = select_fit_validation_masks(
#         valid_anchor, rs_depth_L, fit_samples, validation_ratio,
#         near_sample_weight, rng=rng)
#     if fit_mask is None or fit_mask.sum() < 50:
#         return None, {}

#     common_info = {
#         "anchor_count": int(valid_anchor.sum()),
#         "fit_sample_count": int(fit_mask.sum()),
#     }
#     if da_metric:
#         ratio = float(np.median(rs_depth_L[fit_mask] / da_L[fit_mask]))
#         out = (da_L * ratio).astype(np.float32)
#         out[~np.isfinite(out) | (out <= 0)] = np.nan
#         info = {"mode": "metric", "ratio": ratio, "inl": 1.0,
#                 "inlier_count": int(fit_mask.sum()), **common_info}
#         info.update(validation_depth_statistics(
#             out, rs_depth_L, validation_mask))
#         return out, info

#     inv_gt = np.zeros_like(rs_depth_L)
#     inv_gt[valid_anchor] = 1.0 / rs_depth_L[valid_anchor]
#     fit = robust_affine_invdepth(da_L, inv_gt, fit_mask, rng=rng)
#     if fit is None:
#         return None, {}
#     s, t, inl, inlier_count, fit_sample_count = fit
#     if fit_mode == "isotonic-pchip":
#         curved = isotonic_pchip_invdepth(
#             da_L, inv_gt, fit_mask, metric_depth_max)
#         if curved is None:
#             return None, {}
#         ga_inv, curve_info = curved
#     else:
#         ga_inv = s * da_L + t
#         da_far_threshold = ((1.0 / metric_depth_max - t) / s
#                             if s > 0 else float(np.nanmin(da_L[fit_mask])))
#         curve_info = {
#             "mode": "affine", "s": s, "t": t, "inl": inl,
#             "inlier_count": inlier_count, "fit_sample_count": fit_sample_count,
#             "da_far_threshold": float(da_far_threshold),
#         }
#     good = np.isfinite(ga_inv) & (ga_inv > 0)
#     depth_L = np.full_like(da_L, np.nan, dtype=np.float32)
#     depth_L[good] = (1.0 / ga_inv[good]).astype(np.float32)
#     info = {**curve_info, **common_info,
#             "coverage": float(good.mean())}
#     info.update(validation_depth_statistics(
#         depth_L, rs_depth_L, validation_mask))
#     return depth_L, info


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


def _minimum_text(depth, valid=None):
    mask = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        mask &= valid
    return f"min={float(depth[mask].min()):.2f}m" if mask.any() else "min=N/A"


def _fit_diagnostic(da_L, rs_depth_L, anchor_valid, info, tile):
    """Draw DA-relative vs RealSense inverse-depth and affine candidates."""
    canvas = np.full((tile, tile, 3), 245, dtype=np.uint8)
    x = da_L[anchor_valid].astype(np.float64)
    y = (1.0 / rs_depth_L[anchor_valid]).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return _tile(canvas, "weighted-MSE fit: N/A", tile)

    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])
    xspan, yspan = max(xhi - xlo, 1e-9), max(yhi - ylo, 1e-9)
    px = np.clip(((x - xlo) / xspan * (tile - 25)).astype(int), 0, tile - 1)
    py = np.clip((tile - 1 - (y - ylo) / yspan * (tile - 25)).astype(int),
                 22, tile - 1)

    s, t = info.get("s", np.nan), info.get("t", np.nan)
    if np.isfinite(s) and np.isfinite(t):
        residual = y - (s * x + t)
        center = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - center)) + 1e-9
        inlier = np.abs(residual - center) < 2.5 * mad
    else:
        inlier = np.ones(x.shape, dtype=bool)
    show = np.linspace(0, x.size - 1, min(4000, x.size)).astype(int)
    for index in show:
        color = (40, 150, 40) if inlier[index] else (40, 40, 220)
        canvas[py[index], px[index]] = color

    def draw_line(slope, shift, color, width=1):
        yy0, yy1 = slope * xlo + shift, slope * xhi + shift
        p0 = (0, int(np.clip(tile - 1 - (yy0 - ylo) / yspan * (tile - 25), 22, tile - 1)))
        p1 = (tile - 1, int(np.clip(tile - 1 - (yy1 - ylo) / yspan * (tile - 25), 22, tile - 1)))
        cv2.line(canvas, p0, p1, color, width, cv2.LINE_AA)

    candidate_colors = ((255, 120, 0), (180, 0, 180), (0, 170, 220), (180, 120, 0))
    # fit_candidates are the five lowest-MSE legal hypotheses evaluated by
    # weighted_mse_affine_invdepth(). Index 0 is the selected final line.
    candidates = info.get("fit_candidates", [])
    for candidate, color in zip(candidates[1:5], candidate_colors):
        draw_line(candidate["s"], candidate["t"], color)
    if np.isfinite(s) and np.isfinite(t):
        draw_line(s, t, (0, 0, 0), 2)
    wmse = info.get("weighted_invdepth_mse", np.nan)
    title = f"weighted-MSE fit | mse={wmse:.3g}"
    return _tile(canvas, title, tile)


def _residual_map(depth_L, rs_depth_L, anchor_valid, max_error=2.0):
    valid = (anchor_valid & np.isfinite(depth_L) & np.isfinite(rs_depth_L))
    residual = np.zeros(depth_L.shape, dtype=np.float32)
    residual[valid] = np.abs(depth_L[valid] - rs_depth_L[valid])
    normalized = np.clip(residual / max(float(max_error), 1e-6), 0, 1)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 30
    return color


def _combined_mask_bgr(valid_mask, far_mask, sky_mask, fov_mask):
    # Outside FOV=dark gray, unlabelled=dark, metric=black, far=gray, sky=white.
    out = np.full((*valid_mask.shape, 3), 48, dtype=np.uint8)
    out[~fov_mask] = 24
    out[valid_mask] = 0
    out[far_mask] = 127
    out[sky_mask] = 255
    return out


def build_montage(left_img, da_L, d455_depth_L, d435_depth_L, rs_depth_L,
                  anchor_valid, depth_L, info, fov_mask, gt_valid, disp, tile,
                  dmin, dmax, sky_mask=None, gt_depth=None, far_mask=None,
                  supervised_mask=None):
    left_bgr = _to_bgr(left_img)
    da_rel = _rel_bgr(da_L)
    d455_valid = np.isfinite(d455_depth_L) & (d455_depth_L > 0)
    d435_valid = np.isfinite(d435_depth_L) & (d435_depth_L > 0)
    d455_col = _depth_bgr(d455_depth_L, d455_valid, dmin, dmax)
    d435_col = _depth_bgr(d435_depth_L, d435_valid, dmin, dmax)
    dL = _depth_bgr(depth_L, np.isfinite(depth_L), dmin, dmax)
    gt_depth = depth_L if gt_depth is None else gt_depth
    gt_display_valid = gt_valid if far_mask is None else (gt_valid | far_mask)
    if sky_mask is not None:
        gt_display_valid |= sky_mask
    gtm = _depth_bgr(gt_depth, gt_display_valid, dmin, dmax)
    if disp is None:
        dsp = np.full_like(left_bgr, 30)
        disp_title = "LightStereo: not available"
    else:
        dsp = _rel_bgr(disp, np.isfinite(disp) & (disp > 0))
        disp_title = "LightStereo disparity"
    val_mae = info.get("val_mae_lt5m", np.nan)
    val_mae_txt = "n/a" if not np.isfinite(val_mae) else f"{val_mae:.2f}m"
    far_ratio = 0.0 if far_mask is None else float(far_mask.mean())
    gt_txt = (f"final GT | {_minimum_text(gt_depth, supervised_mask)} "
              f"valid={gt_valid.mean()*100:.0f}% far={far_ratio*100:.0f}%")

    # sky overlay on the left image (cyan = sky)
    if sky_mask is not None and sky_mask.any():
        sky_over = left_bgr.copy()
        sky_over[sky_mask] = (0.4 * sky_over[sky_mask]
                              + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
        sky_txt = f"sky seg ({sky_mask.mean()*100:.0f}%)"
    else:
        sky_over = left_bgr.copy()
        sky_txt = "sky seg (0%)"

    combined_mask = _combined_mask_bgr(
        gt_valid, far_mask, sky_mask, fov_mask)
    overlap = cv2.addWeighted(left_bgr, 0.55, gtm, 0.45, 0)
    residual = _residual_map(depth_L, rs_depth_L, anchor_valid)

    row1 = np.hstack([
        _tile(left_bgr, "camera", tile),
        _tile(d455_col, f"D455 aligned | {_minimum_text(d455_depth_L)}", tile, dmin, dmax),
        _tile(d435_col, f"D435 aligned | {_minimum_text(d435_depth_L)}", tile, dmin, dmax),
        _tile(da_rel, "DA relative", tile),
    ])
    row2 = np.hstack([
        _tile(dL, f"fitted GT | {_minimum_text(depth_L)}", tile, dmin, dmax),
        _tile(combined_mask, "valid=black far=gray sky=white", tile),
        _tile(gtm, gt_txt, tile, dmin, dmax),
        _tile(overlap, "camera + final GT", tile),
    ])
    row3 = np.hstack([
        _fit_diagnostic(da_L, rs_depth_L, anchor_valid, info, tile),
        _tile(residual, f"anchor residual | 0-{2:.0f}m val<5={val_mae_txt}", tile, 0, 2),
        _tile(sky_over, sky_txt, tile),
        _tile(dsp, disp_title, tile),
    ])
    W = max(row1.shape[1], row2.shape[1], row3.shape[1])
    padr = lambda r: np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=20)
    return np.vstack([padr(row1), padr(row2), padr(row3)])


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
    if info.get("mode") in ("affine", "weighted-mse", "isotonic-pchip"):
        if abs(info.get("s", 0.0)) < min_scale:
            reasons.append(f"s={info.get('s', 0.0):.4f}<{min_scale}")
    if info.get("mode") in ("affine", "isotonic-pchip"):
        if info.get("inl", 1.0) < min_inlier:
            reasons.append(f"inl={info.get('inl', 0.0):.2f}<{min_inlier}")
    return bool(reasons), reasons


FIT_STATS_FIELDS = [
    "frame", "source_frame", "stamp", "fit_mode", "scale", "shift",
    "da_far_threshold",
    "anchor_count", "sample_count", "inlier_count", "inlier_ratio",
    "weighted_invdepth_mse", "candidate_count",
    "validation_count", "validation_valid_count",
    "D455_coverage", "D435_coverage", "merged_coverage", "GT_valid_ratio",
    "far_ratio", "supervised_ratio",
    "val_MAE_all_m", "val_RMSE_all_m", "val_median_relative_error",
    "val_MAE_0_2m", "val_MAE_2_5m", "val_MAE_5_10m",
    "val_MAE_10_20m", "depth_p10", "depth_p50", "depth_p90",
    "QC_status", "QC_reason",
]


def initialize_stats_csv(path):
    with open(path, "x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIT_STATS_FIELDS).writeheader()


def append_stats_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIT_STATS_FIELDS).writerow(row)


def depth_percentiles(depth, valid):
    values = depth[valid & np.isfinite(depth)]
    if not values.size:
        return np.nan, np.nan, np.nan
    return tuple(float(value) for value in np.percentile(values, [10, 50, 90]))


def output_filename_prefix(bag_path):
    """Return DATE_SESSION from .../DATE/SESSION/<bag>.mcap."""
    path = Path(bag_path).expanduser()
    folder = path if path.is_dir() else path.parent
    parts = folder.parts
    date_index = next(
        (i for i in range(len(parts) - 1, -1, -1)
         if re.fullmatch(r"\d{4}", parts[i])),
        None,
    )
    if date_index is not None:
        names = [parts[date_index]]
        if date_index + 1 < len(parts):
            names.append(parts[date_index + 1])
    else:
        names = [folder.name]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(names)).strip("_")


def validate_output_prefix(prefix):
    if not prefix or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix) is None:
        raise ValueError(
            "output prefix must contain only letters, digits, '_' and '-'"
        )
    return prefix


def find_existing_outputs(prefix, vis_dir, export_dir, review_subdir):
    checks = [(Path(vis_dir), ".png")]
    if export_dir:
        checks.append((Path(export_dir), ".npz"))
    found = []
    for root, suffix in checks:
        for directory in (root, root / review_subdir):
            found.extend(sorted(directory.glob(f"{prefix}_*{suffix}")))
    return found


def write_png_exclusive(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode PNG: {path}")
    with open(path, "xb") as stream:
        stream.write(encoded.tobytes())


def save_npz_exclusive(path, **arrays):
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)


def _yaml_value(value):
    """Convert argparse/NumPy/Path values to YAML-safe built-in values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return value


def write_resolved_config_exclusive(path, args, source_config,
                                    resolved_calibration):
    """Save the final GT settings without overwriting an earlier run."""
    import yaml
    payload = {
        "source_gt_config": args.gt_config,
        "arguments": _yaml_value(vars(args)),
        "source_config": _yaml_value(source_config),
        "resolved_calibration": _yaml_value(resolved_calibration),
    }
    with open(path, "x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None, processor=None):
    global CAM0_PROJ, CAM1_PROJ, CAM1_DIST, T_LI

    reset_runtime_defaults()
    processor = processor or GTProcessor()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gt-config")
    pre_args, _ = pre.parse_known_args(argv)
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
    ap.add_argument("--gt-config", required=True,
                    help="GT runtime config; not a Kalibr camchain")
    ap.add_argument("--bag", default=config_path(io_cfg.get("bag"), gt_cfg_path))
    ap.add_argument("--vis-dir",
                    default=config_path(io_cfg.get("vis_dir", "./gt_vis"), gt_cfg_path))
    ap.add_argument("--export-dir",
                    default=config_path(io_cfg.get("export_dir"), gt_cfg_path))
    ap.add_argument("--output-prefix", default=io_cfg.get("output_prefix"),
                    help="safe PNG/NPZ stem prefix; defaults to DATE_SESSION")
    ap.add_argument("--stats-csv",
                    default=config_path(io_cfg.get("stats_csv"), gt_cfg_path),
                    help="per-bag CSV path; defaults to VIS_DIR/PREFIX.csv")
    ap.add_argument("--resolved-config", default=None,
                    help=("final resolved YAML path; defaults to EXPORT_DIR/PREFIX_"
                          "resolved_config.yaml, or VIS_DIR when export is disabled"))
    ap.add_argument("--max-pairs", type=int, default=io_cfg.get("max_pairs", 1000))
    ap.add_argument("--frame-interval", "--max-pairs-interval",
                    dest="frame_interval", type=int,
                    default=io_cfg.get("frame_interval", 1),
                    help="run inference on every Nth disparity frame")
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

    configured_deploy_mask = mask_cfg.get(
        "deploy_mask", gt_cfg.get("deploy_mask", "auto"))
    if configured_deploy_mask != "auto":
        configured_deploy_mask = config_path(configured_deploy_mask, gt_cfg_path)
    ap.add_argument(
        "--deploy-mask", default=configured_deploy_mask,
        help=("FOV .npy override; default 'auto' derives pair_<L>_<R>_circle_fov.npy "
              "from --left-topic"),
    )

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
    ap.add_argument("--dmax", type=float, default=runtime_cfg.get("dmax", 20.0))
    ap.add_argument("--max-fit-depth", type=float,
                    default=runtime_cfg.get("max_fit_depth", 20.0))
    ap.add_argument("--metric-depth-max", "--gt-max-depth",
                    dest="metric_depth_max", type=float,
                    default=runtime_cfg.get(
                        "metric_depth_max", runtime_cfg.get("gt_max_depth", 20.0)),
                    help="largest distance supervised as exact metric depth")
    ap.add_argument("--far-depth", type=float,
                    default=runtime_cfg.get("far_depth", 19.0),
                    help="label meaning farther than --metric-depth-max")
    ap.add_argument("--sky-depth", type=float,
                    default=runtime_cfg.get("sky_depth", 20.0),
                    help="separate fixed depth label for sky pixels")
    ap.add_argument("--model-max-depth", type=float,
                    default=runtime_cfg.get("model_max_depth", 20.0),
                    help="expected training-model output ceiling; metadata/safety check")
    ap.add_argument("--fit-mode",
                    choices=["affine", "weighted-mse", "isotonic-pchip"],
                    default=runtime_cfg.get("fit_mode", "weighted-mse"),
                    help="DA-relative to inverse-metric alignment model")
    ap.add_argument("--d435-merge", choices=["fill", "min"],
                    default=runtime_cfg.get(
                        "d435_merge", d435_cfg.get("merge", "min")))
    ap.add_argument("--no-splat", dest="no_splat", action="store_true",
                    default=not runtime_cfg.get("splat", True),
                    help="disable 2x2 splat (leave sub-pixel holes in the warp)")
    ap.add_argument("--splat", dest="no_splat", action="store_false")
    ap.add_argument("--fit-samples", type=int,
                    default=runtime_cfg.get("fit_samples", 15000),
                    help="maximum near-weighted anchors used by RANSAC")
    ap.add_argument("--near-sample-weight", type=float,
                    default=runtime_cfg.get("near_sample_weight", 3.0),
                    help="sampling weight for RealSense anchors at <=5 m")
    ap.add_argument("--ransac-validation-ratio", type=float,
                    default=runtime_cfg.get("ransac_validation_ratio", 0.05),
                    help="near-weighted anchor fraction held out from fitting")

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
    ap.add_argument("--sky-dynamic-input-scale", dest="sky_dynamic_input_scale",
                    action="store_true",
                    default=sky_cfg.get("dynamic_input_scale", True))
    ap.add_argument("--sky-no-dynamic-input-scale",
                    dest="sky_dynamic_input_scale", action="store_false")
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
    args = ap.parse_args(argv)

    if not args.bag:
        ap.error("--bag is required unless io.bag is set in the GT config")
    if args.fit_samples < 50:
        ap.error("--fit-samples must be at least 50")
    if args.frame_interval < 1:
        ap.error("--frame-interval must be at least 1")
    if args.near_sample_weight <= 0:
        ap.error("--near-sample-weight must be positive")
    if not 0.0 <= args.ransac_validation_ratio < 1.0:
        ap.error("--ransac-validation-ratio must be in [0, 1)")
    if args.metric_depth_max <= args.dmin:
        ap.error("--metric-depth-max must be greater than --dmin")
    if args.metric_depth_max > args.model_max_depth:
        ap.error("--metric-depth-max cannot exceed --model-max-depth")
    if not args.dmin < args.far_depth <= args.model_max_depth:
        ap.error("require dmin < far-depth <= model-max-depth")
    if not args.dmin < args.sky_depth <= args.model_max_depth:
        ap.error("require dmin < sky-depth <= model-max-depth")
    if args.deploy_mask in (None, "auto"):
        args.deploy_mask = deploy_mask_from_left_topic(args.left_topic)

    # CLI paths are relative to the current directory; YAML paths were already
    # resolved relative to gt_config.yaml above.
    for attr in ("bag", "vis_dir", "export_dir", "stats_csv", "resolved_config",
                 "deploy_mask",
                 "left_calib", "d435_calib", "sky_param", "sky_bin"):
        value = getattr(args, attr)
        if value:
            setattr(args, attr, str(Path(value).expanduser().resolve()))

    if gt_cfg_path is not None:
        args.gt_config = str(gt_cfg_path)
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
    filename_prefix = validate_output_prefix(
        args.output_prefix or output_filename_prefix(args.bag)
    )
    resolved_config_path = (args.resolved_config or os.path.join(
        args.export_dir or args.vis_dir,
        f"{filename_prefix}_resolved_config.yaml"))
    resolved_config_path = str(Path(resolved_config_path).expanduser().resolve())
    os.makedirs(os.path.dirname(resolved_config_path) or ".", exist_ok=True)
    stats_csv_path = (args.stats_csv or
                      os.path.join(args.vis_dir, f"{filename_prefix}.csv"))
    os.makedirs(os.path.dirname(stats_csv_path) or ".", exist_ok=True)
    existing_outputs = find_existing_outputs(
        filename_prefix, args.vis_dir, args.export_dir, args.qc_subdir)
    if existing_outputs:
        preview = "\n  ".join(str(path) for path in existing_outputs[:5])
        raise SystemExit(
            "Refusing to overwrite existing output(s):\n  " + preview
        )
    if os.path.exists(stats_csv_path):
        raise SystemExit(f"Refusing to overwrite existing CSV: {stats_csv_path}")
    if os.path.exists(resolved_config_path):
        raise SystemExit(
            f"Refusing to overwrite resolved config: {resolved_config_path}")
    print(f"Output filename prefix: {filename_prefix}")
    initialize_stats_csv(stats_csv_path)
    print(f"  fitting statistics -> {stats_csv_path}")

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
    with AnyReader([Path(args.bag)], default_typestore=TYPESTORE) as reader:
        available_topics = {connection.topic for connection in reader.connections}
    disparity_available = args.disp_topic in available_topics
    anchor_topic = args.disp_topic if disparity_available else args.left_topic
    if not disparity_available:
        print(f"  optional LightStereo topic '{args.disp_topic}' is absent; "
              "using left-image timestamps and skipping its montage panel")

    print("reading bag...")
    data, scanned_counts = load_topics_sampled(
        args.bag,
        topics,
        anchor_topic,
        args.frame_interval,
        args.sync_tol,
        {
            args.left_topic: 0.0,
            args.d455_depth: -args.depth_time_offset,
            args.d435_depth: -args.d435_time_offset,
        },
        {args.d455_info, args.d435_info},
    )
    for t in topics:
        print(f"  {t}: scanned={scanned_counts[t]} retained={len(data[t])}")

    for required_topic in (args.d455_depth, args.left_topic):
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
    proc, model = processor.load_da_v2(args.da_model, device)

    out_hw = image_to_numpy(data[args.left_topic][0][1]).shape[:2]
    print(f"  stereo-left size (HxW) = {out_hw}  device={device}")

    deploy_mask = None
    if args.deploy_mask:
        deploy_mask = load_deploy_mask(args.deploy_mask, out_hw)
        print(f"  deploy mask {args.deploy_mask}: coverage {100*deploy_mask.mean():.1f}%")
    else:
        raise RuntimeError("circle/wing FOV mask is required")

    # sky segmentation
    sky_seg = None
    if not args.sky_enabled:
        print("  sky: DISABLED")
    elif args.sky_heuristic:
        print("  sky: brightness/texture heuristic")
    elif args.sky_param and args.sky_bin:
        sky_seg = processor.load_sky(args)
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

    resolved_calibration = {
        "output_hw": list(out_hw),
        "left_intrinsics": CAM1_PROJ,
        "left_distortion": CAM1_DIST,
        "d455_intrinsics": K_I,
        "T_left_from_d455": T_LI,
        "d435_enabled": d435_enabled,
        "d435_intrinsics": K_d435 if d435_enabled else None,
        "d435_distortion": d435_src_dist if d435_enabled else None,
        "T_left_from_d435": T_L_D435 if d435_enabled else None,
    }
    write_resolved_config_exclusive(
        resolved_config_path, args, gt_cfg, resolved_calibration)
    print(f"  resolved config -> {resolved_config_path}")

    n = 0
    n_suspect = 0
    n_skipped_sync = 0
    n_skipped_anchors = 0
    n_skipped_fit = 0
    selected_frames = 0
    for source_frame, (t_anchor, anchor_msg) in enumerate(data[anchor_topic]):
        if n >= args.max_pairs:
            break
        selected_frames += 1
        dm = nearest(data[args.d455_depth],
                     t_anchor - args.depth_time_offset, args.sync_tol)
        lm = nearest(data[args.left_topic], t_anchor, args.sync_tol)
        if dm is None or lm is None:
            if n_skipped_sync < 5:
                print(f"  [skip @ anchor t={t_anchor:.3f}] "
                      f"depth={'MISS' if dm is None else 'ok'} "
                      f"left={'MISS' if lm is None else 'ok'}")
            n_skipped_sync += 1
            continue

        depth_I = image_to_numpy(dm[1]).astype(np.float32) * args.depth_scale   # infra1 frame
        left = image_to_numpy(lm[1])
        disp_L = None
        if disparity_available:
            disp = image_to_numpy(anchor_msg).astype(np.float32)
            if disp.ndim == 2:
                disp_L = disp
                if disp.shape != out_hw:
                    disp_L = cv2.resize(
                        disp, (out_hw[1], out_hw[0]),
                        interpolation=cv2.INTER_NEAREST)
            elif source_frame == 0:
                print("  LightStereo disparity is not scalar 32FC1; "
                      "skipping its montage panel")

        # ---- STEP 1 ----
        da_L = step1_da_on_left(proc, model, left, device)
        # ---- STEP 2 (D455 infra1 depth -> left, single Kalibr hop) ----
        d455_depth_L = step2_ir_depth_to_L(
            depth_I, K_I, T_LI, CAM1_PROJ, CAM1_DIST,
            out_hw, splat=not args.no_splat)
        rs_depth_L = d455_depth_L.copy()
        cover455 = float(np.isfinite(d455_depth_L).mean())

        # ---- STEP 2b (D435 depth -> left) merged in ----
        cover435 = 0.0
        d435_depth_L = np.full(out_hw, np.nan, dtype=np.float32)
        if d435_enabled:
            d435m = nearest(data[args.d435_depth],
                            t_anchor - args.d435_time_offset, args.sync_tol)
            if d435m is not None:
                depth_435 = image_to_numpy(d435m[1]).astype(np.float32) * args.d435_depth_scale
                d435_depth_L = step2_ir_depth_to_L(
                    depth_435, K_d435, T_L_D435, CAM1_PROJ, CAM1_DIST, out_hw,
                    splat=not args.no_splat, src_dist=d435_src_dist)
                cover435 = float(np.isfinite(d435_depth_L).mean())
                rs_depth_L = merge_depth_L(
                    rs_depth_L, d435_depth_L, mode=args.d435_merge)

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
                                           args.da_metric,
                                           fit_mode=args.fit_mode,
                                           metric_depth_max=args.metric_depth_max,
                                           fit_samples=args.fit_samples,
                                           near_sample_weight=args.near_sample_weight,
                                           validation_ratio=args.ransac_validation_ratio)
        if depth_L is None:
            if n_skipped_fit < 5:
                print(f"  [skip @ disp t={t_anchor:.3f}] fit returned None "
                      f"(anchors={int(anchor_valid.sum())})")
            n_skipped_fit += 1
            continue
        # ---- STEP 4 ----
        # Exact metric supervision stops at metric_depth_max. Low raw relative-DA
        # values below the 15 m curve threshold are far even when PCHIP clamps at
        # its lowest supported anchor instead of extrapolating.
        far_from_da = np.zeros(out_hw, dtype=bool)
        da_far_threshold = info.get("da_far_threshold", np.nan)
        if not args.da_metric and np.isfinite(da_far_threshold):
            far_from_da = np.isfinite(da_L) & (da_L <= da_far_threshold)
        far_mask = (fov & ~sky_mask & (far_from_da |
                    (np.isfinite(depth_L) &
                     (depth_L > args.metric_depth_max))))
        metric_valid = (fov & np.isfinite(depth_L) & (depth_L > args.dmin)
                        & (depth_L <= args.metric_depth_max)
                        & ~sky_mask & ~far_mask)
        supervised_mask = metric_valid | far_mask | sky_mask
        gt_final = np.zeros(out_hw, dtype=np.float32)
        gt_final[metric_valid] = depth_L[metric_valid]
        gt_final[far_mask] = np.float32(args.far_depth)
        gt_final[sky_mask] = np.float32(args.sky_depth)

        suspect, reasons = (False, [])
        if args.qc_enabled:
            suspect, reasons = fit_is_suspect(
                depth_L, metric_valid, info, args.qc_min_spread,
                args.qc_min_s, args.qc_min_inl)
        info["qc_status"] = "REVIEW" if suspect else "PASS"
        info["qc_reason"] = ";".join(reasons)
        vis_dir = review_vis if suspect else args.vis_dir
        export_dir = review_export if suspect else args.export_dir

        p10, p50, p90 = depth_percentiles(depth_L, metric_valid)
        append_stats_csv(stats_csv_path, {
            "frame": n + 1,
            "source_frame": source_frame,
            "stamp": t_anchor,
            "fit_mode": info.get("mode", args.fit_mode),
            "scale": info.get("s", np.nan),
            "shift": info.get("t", np.nan),
            "da_far_threshold": info.get("da_far_threshold", np.nan),
            "anchor_count": info.get("anchor_count", 0),
            "sample_count": info.get("fit_sample_count", 0),
            "inlier_count": info.get("inlier_count", 0),
            "inlier_ratio": info.get("inl", np.nan),
            "weighted_invdepth_mse": info.get(
                "weighted_invdepth_mse", np.nan),
            "candidate_count": info.get("candidate_count", 0),
            "validation_count": info.get("validation_count", 0),
            "validation_valid_count": info.get("validation_valid_count", 0),
            "D455_coverage": cover455,
            "D435_coverage": cover435,
            "merged_coverage": float(np.isfinite(rs_depth_L).mean()),
            "GT_valid_ratio": float(metric_valid.mean()),
            "far_ratio": float(far_mask.mean()),
            "supervised_ratio": float(supervised_mask.mean()),
            "val_MAE_all_m": info.get("val_mae_m", np.nan),
            "val_RMSE_all_m": info.get("val_rmse_m", np.nan),
            "val_median_relative_error": info.get(
                "val_median_relative_error", np.nan),
            "val_MAE_0_2m": info.get("val_mae_0_2m", np.nan),
            "val_MAE_2_5m": info.get("val_mae_2_5m", np.nan),
            "val_MAE_5_10m": info.get("val_mae_5_10m", np.nan),
            "val_MAE_10_20m": info.get("val_mae_10_20m", np.nan),
            "depth_p10": p10,
            "depth_p50": p50,
            "depth_p90": p90,
            "QC_status": "REVIEW" if suspect else "PASS",
            "QC_reason": ";".join(reasons),
        })

        # ---- STEP 6 (viz) ----
        montage = build_montage(left, da_L, d455_depth_L, d435_depth_L,
                                rs_depth_L, anchor_valid, depth_L, info,
                                fov, metric_valid, disp_L,
                                args.tile, args.dmin, args.dmax, sky_mask=sky_mask,
                                gt_depth=gt_final, far_mask=far_mask,
                                supervised_mask=supervised_mask)
        output_stem = f"{filename_prefix}_{n + 1}"
        write_png_exclusive(
            os.path.join(vis_dir, f"{output_stem}.png"), montage)

        # ---- STEP 5 (export) ----
        if export_dir:
            save_npz_exclusive(
                os.path.join(export_dir, f"{output_stem}.npz"),
                # disp=disp.astype(np.float32),
                depth_aligned=gt_final,
                realsense_depth_d455_aligned=d455_depth_L.astype(np.float32),
                realsense_depth_d435_aligned=d435_depth_L.astype(np.float32),
                realsense_depth_merged=rs_depth_L.astype(np.float32),
                left=left,
                valid_mask=metric_valid.astype(np.bool_),
                metric_valid_mask=metric_valid.astype(np.bool_),
                far_mask=far_mask.astype(np.bool_),
                sky_mask=(sky_mask & fov).astype(np.bool_),
                da_relative=da_L.astype(np.float32),
                metric_depth_max=np.float32(args.metric_depth_max),
                far_depth=np.float32(args.far_depth),
                sky_depth=np.float32(args.sky_depth),
                model_max_depth=np.float32(args.model_max_depth),
                # has_disp=np.bool_(True),
                stamp=np.float64(t_anchor),
            )

        if suspect:
            n_suspect += 1
            print(f"  [review] frame {n + 1} -> {args.qc_subdir}/ "
                  f"({', '.join(reasons)})")

        if n % 20 == 0:
            fitmsg = (f"s={info['s']:.2f} "
                      f"wmse={info.get('weighted_invdepth_mse', np.nan):.3g}"
                      if info.get("mode") == "weighted-mse" else
                      (f"s={info['s']:.2f} inl={info['inl']*100:.0f}%")
                      if info.get("mode") == "affine" else
                      (f"pchip={info.get('pchip_knots', 0)} "
                       f"inl={info.get('inl', 0)*100:.0f}%")
                      if info.get("mode") == "isotonic-pchip" else
                      f"x{info.get('ratio',1):.2f}")
            cov_txt = (f"cover D455={cover455*100:.0f}% D435={cover435*100:.0f}% "
                       f"merged={np.isfinite(rs_depth_L).mean()*100:.0f}%"
                       if d435_enabled else
                       f"RS->L cover={np.isfinite(rs_depth_L).mean()*100:.0f}%")
            print(f"  frame {n + 1}: {cov_txt}  anchors={int(anchor_valid.sum())} "
                  f"samples={info.get('fit_sample_count', 0)}  "
                      f"fit[{fitmsg}] metric={metric_valid.mean()*100:.0f}% "
                      f"far={far_mask.mean()*100:.0f}%")
        n += 1

    print(f"done. {n} frames -> {args.vis_dir}"
          + (f" and {args.export_dir}" if args.export_dir else ""))
    print(f"SKIP SUMMARY (of {selected_frames} selected disparity frames, "
          f"interval={args.frame_interval}):")
    print(f"  sync-miss   : {n_skipped_sync}")
    print(f"  <50 anchors : {n_skipped_anchors}")
    print(f"  fit failed  : {n_skipped_fit}")
    print(f"  kept        : {n}")
    if args.qc_enabled:
        print(f"QC: {n - n_suspect} kept, {n_suspect} moved to "
              f"{args.qc_subdir}/")
    return {
        "kept": n,
        "review": n_suspect,
        "sync_miss": n_skipped_sync,
        "anchor_skip": n_skipped_anchors,
        "fit_skip": n_skipped_fit,
    }


if __name__ == "__main__":
    main()
