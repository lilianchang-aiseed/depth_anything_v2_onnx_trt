#!/usr/bin/env python3
"""ROS 2 stereo disparity node for Jetson Orin.

Subscribes (approximate-time-synchronized):
  /camera_2/image_raw  →  left image  (bgr8)
  /camera_1/image_raw  →  right image (bgr8)

Publishes:
  /stereo/left/image_rect   sensor_msgs/Image  bgr8    rectified left
  /stereo/right/image_rect  sensor_msgs/Image  bgr8    rectified right
  /stereo/disparity         sensor_msgs/Image  32FC1   raw disparity (pixels)

Rectification uses the Double Sphere model from stereo_calib_ds2-camchain.yaml.
Disparity is inferred by LightStereo-L (finetune checkpoint).
"""

import os
import sys
import types

import numpy as np
import cv2
import yaml
import torch
import torch.nn.functional as F
from easydict import EasyDict

import rclpy
from rclpy.node import Node
import rclpy.qos
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
_RECTIFY_DIR = os.path.join(_HERE, 'rectify')
_OMNI_DIR    = os.path.join(_HERE, 'omniVidar')
_OPENSTEREO  = os.path.join(_OMNI_DIR, 'OpenStereo')

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
    epiploar_planes_from_extrinsics,
    rasterize_points_map,
    rasterize_points_to_image,
    shape_to_pixel_grid,
)
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from stereo.datasets.dataset_utils.stereo_trans import (
    DivisiblePad, TransposeImage, ToTensor, NormalizeImage, Compose,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CALIB_FILE = os.path.join(_RECTIFY_DIR, 'stereo_calib_ds2-camchain.yaml')
CKPT_FILE  = os.path.join(_OMNI_DIR, 'output',
                          'finetune_lightstereo_l_0421_v10',
                          'best_model_epoch_228_epe_0.0683.pth')
# CKPT_FILE  =  ('/home/nvidia/ros_stereo/omniVidar/output/finetune_lightstereo_l_0421_v10/LightStereo-S-SceneFlow.ckpt')
CFG_FILE   = os.path.join(_OPENSTEREO, 'cfgs',
                          'lightstereo', 'lightstereo_l_sceneflow_general.yaml')

TRT_ENGINE_FILE = "/home/nvidia/ros_stereo/lightstereo_s.engine"

OUTPUT_SIZE    = 320
CONF_THRESHOLD = 0.3   # pixels with conf < threshold are zeroed in disp_pred
CUT_ANGLE      = np.pi / 4
ANGLE_EPS      = np.cos(np.pi / 2 - CUT_ANGLE)   # ≈ 0.5


# ---------------------------------------------------------------------------
# Helpers (model)
# ---------------------------------------------------------------------------
def _ds_intrinsics(cam_cfg, s=1.):
    """Unpack [xi, alpha, fx, fy, cx, cy] from cam config intrinsics list."""
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
    """Add sinusoidal dot pattern to textureless regions to aid stereo matching.

    img: (C, H, W) float32 tensor normalized to [0, 1]
    Returns: same shape, clamped to [0, 1]
    """
    _, H, W = img.shape
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
    grad_mag = torch.sqrt(F.conv2d(gray, sobel_x, padding=1) ** 2 +
                          F.conv2d(gray, sobel_y, padding=1) ** 2)
    mask = (grad_mag < threshold).float()   # (1, 1, H, W)
    xs = torch.linspace(0, 2 * torch.pi * freq, W, device=img.device, dtype=img.dtype)
    ys = torch.linspace(0, 2 * torch.pi * freq, H, device=img.device, dtype=img.dtype)
    pattern = (torch.sin(xs).view(1, W) * torch.sin(ys).view(H, 1)).unsqueeze(0)
    return torch.clamp(img + sigma * mask.squeeze(0) * pattern, 0, 1)


class _TRTEngine:
    """TensorRT 10.x inference using PyTorch CUDA tensors.

    Avoids pycuda entirely — uses PyTorch's active CUDA context and stream so
    there is no context-mismatch between torch and the TRT execution context.
    """

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

        # Allocate contiguous CUDA tensors via PyTorch (shares its CUDA context).
        # Register addresses once — valid for the lifetime of the tensors.
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
        """left, right: float32 (1, 3, H, W) normalised.
        Returns (disp, conf): both (H, W) float32.
        conf is None when the engine was built without confidence output."""
        self._gpu['left_img'].copy_(torch.from_numpy(left),  non_blocking=True)
        self._gpu['right_img'].copy_(torch.from_numpy(right), non_blocking=True)
        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        disp = self._gpu['disp_pred'].cpu().float().numpy().squeeze()  # → (H, W)
        conf = (self._gpu['conf'].cpu().float().numpy().squeeze()
                if 'conf' in self._gpu else None)
        return disp, conf
# @torch.no_grad()
# def _infer(model: torch.nn.Module, transform: Compose,
#            left_rgb: np.ndarray, right_rgb: np.ndarray,
#            device: torch.device, amp: bool) -> np.ndarray:
#     """Run LightStereo on uint8 RGB arrays; returns disparity map (H, W)."""
#     orig_h, orig_w = left_rgb.shape[:2]

#     # Apply targeted pattern on CPU before the model transform
#     left_t  = torch.from_numpy(left_rgb.astype(np.float32)  / 255.0).permute(2, 0, 1)
#     right_t = torch.from_numpy(right_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
#     left_t  = add_targeted_pattern(left_t)
#     right_t = add_targeted_pattern(right_t)
#     left_f  = left_t.permute(1, 2, 0).numpy() * 255.0
#     right_f = right_t.permute(1, 2, 0).numpy() * 255.0

#     sample = {'left': left_f, 'right': right_f}
#     sample = transform(sample)
#     pad_top, _pad_right, _pad_bottom, pad_left = sample.get('pad', [0, 0, 0, 0])
#     sample['left']  = sample['left'].unsqueeze(0).to(device)
#     sample['right'] = sample['right'].unsqueeze(0).to(device)
#     with torch.cuda.amp.autocast(enabled=amp):
#         pred = model(sample)
#     disp = pred['disp_pred'].squeeze().cpu().float().numpy()
#     return disp[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]
def _infer(engine: _TRTEngine, transform: Compose,
           left_rgb: np.ndarray, right_rgb: np.ndarray):
    """Pre-process → TRT inference → crop.
    Returns (disp, conf): both (H, W) float32. conf is None for older engines."""
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

    left_np  = sample['left'].unsqueeze(0).float().numpy()   # (1,3,H,W)
    right_np = sample['right'].unsqueeze(0).float().numpy()

    disp, conf = engine(left_np, right_np)
    disp = disp[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]
    if conf is not None:
        conf = conf[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]
    return disp, conf



# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class StereoDisparityNode(Node):
    """Rectify camera_2 (left) + camera_1 (right) and infer disparity."""

    def __init__(self):
        super().__init__('stereo_disparity')
        self._bridge = CvBridge()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._amp    = (self._device.type == 'cuda')

        self.get_logger().info('Pre-computing rectification maps from calibration...')
        self._build_rect_maps()

        self.get_logger().info('Loading LightStereo model...')
        self._build_model()

        # Publishers
        self._pub_left  = self.create_publisher(Image, '/stereo/left/image_rect',  1)
        self._pub_right = self.create_publisher(Image, '/stereo/right/image_rect', 1)
        self._pub_disp  = self.create_publisher(Image, '/stereo/disparity',        1)

        # Synchronized subscribers — camera_2 = left, camera_1 = right
        _qos = rclpy.qos.QoSProfile(depth=2)
        sub_left  = message_filters.Subscriber(
            self, Image, '/camera_2/image_raw', qos_profile=_qos)
        sub_right = message_filters.Subscriber(
            self, Image, '/camera_1/image_raw', qos_profile=_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [sub_left, sub_right], queue_size=2, slop=0.05)
        self._sync.registerCallback(self._callback)

        self.get_logger().info(
            f'Ready — device={self._device}, AMP={self._amp}, '
            f'output_size={OUTPUT_SIZE}')

    # ------------------------------------------------------------------
    def _build_rect_maps(self):
        """Pre-compute scatter maps for both cameras (constant since vel=0)."""
        with open(CALIB_FILE) as f:
            calib = yaml.safe_load(f)

        # cam0 ↔ left (camera_2), cam1 ↔ right (camera_1)
        cam0_cfg = calib['cam0']
        cam1_cfg = calib['cam1']
        

        # Image shape from calibration resolution [width, height]
        cW, cH = cam0_cfg['resolution']
        W, H = calib.get('stream_resolution', (cW, cH))
        scale_calib = H/cH
        image_shape = (H, W)
        # self._calib_shape = image_shape  # (H, W) — saved for runtime resize
        self.get_logger().info(f'  Calibration image shape: {image_shape}')

        # Unproject every pixel to a 3D ray for each camera
        pixels = shape_to_pixel_grid(image_shape, stride=1)
        self.get_logger().info('  Unprojecting cam0 rays...')
        xi, alpha, fx, fy, cx, cy = _ds_intrinsics(cam0_cfg, scale_calib)
        rays0, _ = unproject_double_sphere_pixels(pixels, xi, alpha, fx, fy, cx, cy)

        self.get_logger().info('  Unprojecting cam1 rays...')
        xi, alpha, fx, fy, cx, cy = _ds_intrinsics(cam1_cfg, scale_calib)
        rays1, _ = unproject_double_sphere_pixels(pixels, xi, alpha, fx, fy, cx, cy)

        # Extrinsics: T_cn_cnm1 is cam1 expressed in cam0 frame
        T = np.array(cam1_cfg['T_cn_cnm1'], dtype=np.float64)
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)

        # Epipolar plane normals (velocity = 0 → use static baseline only)
        n0, n1 = epiploar_planes_from_extrinsics(R, t)
        n0 = n0.astype(np.float32)
        n1 = n1.astype(np.float32)

        # Pre-compute rasterisation maps — these are reused every frame
        self.get_logger().info('  Building rasterisation maps...')
        map0, coord = rasterize_points_map(
            n0, rays0.astype(np.float32), t, n=OUTPUT_SIZE, eps=ANGLE_EPS)
        cam1_coord = [R.T @ e for e in coord]
        map1, _ = rasterize_points_map(
            n1, rays1.astype(np.float32), t, n=OUTPUT_SIZE, eps=ANGLE_EPS,
            coord=cam1_coord)

        self._map0 = map0   # (points, grid, valid)
        self._map1 = map1
        self.get_logger().info('  Rectification maps ready.')

    # ------------------------------------------------------------------
    # def _build_model(self):
    #     with open(CFG_FILE) as f:
    #         cfgs = EasyDict(yaml.safe_load(f))
    #     cfgs.MODEL.PRETRAINED_MODEL = ''

    #     model = LightStereo(cfgs.MODEL)

    #     self.get_logger().info(f'  Loading checkpoint: {CKPT_FILE}')
    #     ckpt = torch.load(CKPT_FILE, map_location='cpu', weights_only=False)
    #     state = ckpt.get('model_state', ckpt.get('model', ckpt))
    #     if any(k.startswith('module.') for k in state):
    #         state = {k.replace('module.', '', 1): v for k, v in state.items()}
    #     missing, unexpected = model.load_state_dict(state, strict=False)
    #     if missing:
    #         self.get_logger().warn(f'  Missing keys: {missing[:3]}...')
    #     if unexpected:
    #         self.get_logger().warn(f'  Unexpected keys: {unexpected[:3]}...')

    #     self._model     = model.to(self._device).eval()
    #     self._transform = _build_transform()
    #     self.get_logger().info('  LightStereo loaded.')
    
    def _build_model(self):
        self.get_logger().info(f'  Loading TRT engine: {TRT_ENGINE_FILE}')
        self._trt     = _TRTEngine(TRT_ENGINE_FILE)
        self._transform = _build_transform()
        self.get_logger().info('  TensorRT engine loaded.')

    # ------------------------------------------------------------------
    def _callback(self, left_msg: Image, right_msg: Image):
        t0 = self.get_clock().now()

        # Decode BGR images
        left_bgr  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_bgr = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        # Rectify — output is (OUTPUT_SIZE, OUTPUT_SIZE, 3) uint8 BGR
        pts0, grid0, valid0 = self._map0
        pts1, grid1, valid1 = self._map1
        
        rect_left,  _ = rasterize_points_to_image(
            pts0, grid0, valid0, left_bgr,  n=OUTPUT_SIZE, output_type='rgb')
        rect_right, _ = rasterize_points_to_image(
            pts1, grid1, valid1, right_bgr, n=OUTPUT_SIZE, output_type='rgb')

        # Convert BGR → RGB for model (matches PIL.Image.open convention used in infer_stereo.py)
        left_rgb  = cv2.cvtColor(rect_left,  cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB)
        t1 = self.get_clock().now()

        # Disparity inference
        # disp = _infer(self._model, self._transform,
        #               left_rgb, right_rgb, self._device, self._amp)
        disp, conf = _infer(self._trt, self._transform, left_rgb, right_rgb) 
        if conf is not None:
            disp = np.where(conf > CONF_THRESHOLD, disp, 0.0).astype(np.float32)

        # Publish
        stamp    = left_msg.header.stamp
        frame_id = left_msg.header.frame_id

        left_out = self._bridge.cv2_to_imgmsg(rect_left, encoding='bgr8')
        left_out.header.stamp    = stamp
        left_out.header.frame_id = frame_id
        self._pub_left.publish(left_out)

        right_out = self._bridge.cv2_to_imgmsg(rect_right, encoding='bgr8')
        right_out.header.stamp    = stamp
        right_out.header.frame_id = frame_id
        self._pub_right.publish(right_out)

        disp_out = self._bridge.cv2_to_imgmsg(disp, encoding='32FC1')
        disp_out.header.stamp    = stamp
        disp_out.header.frame_id = frame_id
        self._pub_disp.publish(disp_out)

        rect_ms = (t1 - t0).nanoseconds / 1e6
        infer_ms = (self.get_clock().now() - t1).nanoseconds / 1e6
        self.get_logger().info(
            f'disp [{disp.min():.1f}, {disp.max():.1f}] px  | rect {rect_ms:.0f} ms | infer {infer_ms:.0f} ms',
            throttle_duration_sec=1.0)


# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = StereoDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
