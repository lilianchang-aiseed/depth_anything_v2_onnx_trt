#!/usr/bin/env python3
"""Four-camera fisheye Depth Anything V2 TensorRT ROS 2 node.

Subscribe to /camera_{0..3}/image_raw and publish a 504x280 32FC1 relative
depth image on /camera_{id}/relative_depth. Inputs not already 504x280 are
resized first. Four latest-frame slots and one round-robin GPU worker avoid
stale queues and camera starvation.
"""

import argparse
from pathlib import Path
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


NODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = NODE_DIR.parent
DEFAULT_ENGINE = (
    PROJECT_DIR / "checkpoints" / "depth_anything_v2_vits_dynamic.engine"
)

CAMERA_IDS = (0, 1, 2, 3)
INPUT_WIDTH = 504
INPUT_HEIGHT = 280


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
            f"{sorted(conversions)}"
        )

    channels, conversion = conversions[encoding]
    row_bytes = int(message.width) * channels
    if int(message.step) < row_bytes:
        raise ValueError(
            f"Invalid Image step {message.step} for {message.width}x{channels}"
        )
    required_bytes = int(message.step) * int(message.height)
    buffer = np.frombuffer(message.data, dtype=np.uint8)
    if buffer.size < required_bytes:
        raise ValueError(
            f"Image buffer has {buffer.size} bytes; expected {required_bytes}"
        )

    rows = buffer[:required_bytes].reshape(message.height, message.step)
    pixels = rows[:, :row_bytes]
    if channels == 1:
        image = pixels.reshape(message.height, message.width)
    else:
        image = pixels.reshape(message.height, message.width, channels)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return np.ascontiguousarray(image)


def _prepare_image(image_bgr: np.ndarray):
    resized = image_bgr.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH)
    if resized:
        image_bgr = cv2.resize(
            image_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
        )
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    return tensor, resized


class TensorRTEngine:
    """TensorRT 10 engine runner using PyTorch CUDA buffers."""

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
            raise RuntimeError(
                "TensorRT Python bindings are not installed"
            ) from exc

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

        self._input_names = []
        self._output_names = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)
        if len(self._input_names) != 1 or len(self._output_names) != 1:
            raise RuntimeError(
                "Expected one TensorRT input and one output; got "
                f"inputs={self._input_names}, outputs={self._output_names}"
            )
        self.input_name = self._input_names[0]
        self.output_name = self._output_names[0]
        self._buffers = {}

    def infer(self, model_input: np.ndarray) -> np.ndarray:
        input_shape = tuple(model_input.shape)
        accepted = self._context.set_input_shape(self.input_name, input_shape)
        if accepted is False:
            raise ValueError(
                f"TensorRT profile rejected input shape {input_shape}"
            )
        if hasattr(self._context, "infer_shapes"):
            missing = self._context.infer_shapes()
            if missing:
                raise RuntimeError(f"TensorRT shapes remain unspecified: {missing}")

        for name in self._input_names + self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT shape for {name}: {shape}")
            np_dtype = np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
            torch_dtype = self._DTYPE_MAP.get(np_dtype)
            if torch_dtype is None:
                raise TypeError(f"Unsupported TensorRT dtype for {name}: {np_dtype}")
            buffer = self._buffers.get(name)
            if buffer is None or tuple(buffer.shape) != shape or buffer.dtype != torch_dtype:
                buffer = torch.empty(
                    shape, dtype=torch_dtype, device="cuda"
                ).contiguous()
                self._buffers[name] = buffer
            self._context.set_tensor_address(name, buffer.data_ptr())

        input_buffer = self._buffers[self.input_name]
        input_buffer.copy_(torch.from_numpy(model_input), non_blocking=True)
        stream = torch.cuda.current_stream()
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return self._buffers[self.output_name].float().cpu().numpy()


class DepthAnythingV2TensorRTNode(Node):
    def __init__(self, engine_path: Path):
        super().__init__("depth_anything_v2_fisheye_trt")
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        self._engine = TensorRTEngine(engine_path)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._keys = list(CAMERA_IDS)
        # Do not use Node's reserved _publishers/_subscriptions attribute names.
        self._depth_publishers = {}
        self._image_subscriptions = []
        self._latest_frames = {key: None for key in self._keys}
        self._slot_condition = threading.Condition()
        self._stop_worker = False
        self._round_robin_index = 0
        self._last_statistics_log = 0.0

        for camera_id in self._keys:
            input_topic = f"/camera_{camera_id}/image_raw"
            output_topic = f"/camera_{camera_id}/relative_depth"
            key = camera_id
            self._depth_publishers[key] = self.create_publisher(
                Image, output_topic, 1
            )
            subscription = self.create_subscription(
                Image,
                input_topic,
                lambda message, slot_key=key: self._image_callback(
                    slot_key, message
                ),
                qos,
            )
            self._image_subscriptions.append(subscription)
            self.get_logger().info(f"  {input_topic} -> {output_topic}")

        self._worker = threading.Thread(
            target=self._gpu_worker,
            name="depth_anything_v2_gpu_worker",
            daemon=True,
        )
        self._worker.start()
        self.get_logger().info(
            f"Ready: {len(self._keys)} fisheye inputs at "
            f"{INPUT_WIDTH}x{INPUT_HEIGHT}; "
            f"TensorRT input={self._engine.input_name}, "
            f"output={self._engine.output_name}"
        )

    def _image_callback(self, key, message: Image):
        """Replace this input's pending frame instead of building a backlog."""
        with self._slot_condition:
            self._latest_frames[key] = message
            self._slot_condition.notify()

    def _next_frame(self):
        """Return the next pending frame fairly across all eight input slots."""
        with self._slot_condition:
            while not self._stop_worker and not any(
                message is not None for message in self._latest_frames.values()
            ):
                self._slot_condition.wait()
            if self._stop_worker:
                return None, None

            for offset in range(len(self._keys)):
                index = (self._round_robin_index + offset) % len(self._keys)
                key = self._keys[index]
                message = self._latest_frames[key]
                if message is not None:
                    self._latest_frames[key] = None
                    self._round_robin_index = (index + 1) % len(self._keys)
                    return key, message
        return None, None

    def _gpu_worker(self):
        while True:
            key, message = self._next_frame()
            if message is None:
                return
            self._infer_and_publish(key, message)

    def _infer_and_publish(self, key, message: Image):
        try:
            total_started = time.perf_counter()
            image_bgr = _image_msg_to_bgr(message)
            model_input, resized = _prepare_image(image_bgr)
            inference_started = time.perf_counter()
            output = self._engine.infer(model_input)
            depth = np.asarray(output, dtype=np.float32).squeeze()
            expected_shape = (INPUT_HEIGHT, INPUT_WIDTH)
            if depth.shape != expected_shape:
                raise ValueError(
                    f"Unexpected depth shape {depth.shape}; expected {expected_shape}"
                )
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            inference_finished = time.perf_counter()
            self._publish_image(
                depth, "32FC1", message, self._depth_publishers[key]
            )
            finished = time.perf_counter()

            prep_ms = (inference_started - total_started) * 1000.0
            infer_ms = (inference_finished - inference_started) * 1000.0
            publish_ms = (finished - inference_finished) * 1000.0
            total_ms = (finished - total_started) * 1000.0
            now = time.monotonic()
            if now - self._last_statistics_log >= 2.0:
                finite = depth[np.isfinite(depth)]
                if finite.size:
                    self.get_logger().info(
                        f"camera_{key} depth: resized={'yes' if resized else 'no'}, "
                        f"min={finite.min():.6f}, max={finite.max():.6f}, "
                        f"mean={finite.mean():.6f}, nonzero={np.count_nonzero(finite)} "
                        f"| prep={prep_ms:.2f} ms, infer={infer_ms:.2f} ms, "
                        f"pub={publish_ms:.2f} ms, total={total_ms:.2f} ms"
                    )
                else:
                    self.get_logger().error(
                        f"camera_{key} output has no finite values"
                    )
                self._last_statistics_log = now
        except Exception as exc:
            self.get_logger().error(
                f"Depth Anything failed for camera_{key}: {exc}"
            )

    def destroy_node(self):
        with self._slot_condition:
            self._stop_worker = True
            self._slot_condition.notify_all()
        if hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)
        return super().destroy_node()

    @staticmethod
    def _publish_image(array, encoding, source_message, publisher):
        array = np.ascontiguousarray(array)
        message = Image()
        message.header.stamp = source_message.header.stamp
        message.header.frame_id = source_message.header.frame_id
        message.height = array.shape[0]
        message.width = array.shape[1]
        message.encoding = encoding
        message.is_bigendian = int(sys.byteorder == "big")
        message.step = array.strides[0]
        message.data = array.tobytes()
        publisher.publish(message)


def _parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="ROS 2 Depth Anything V2 TensorRT relative-depth node"
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE,
        help=f"TensorRT engine path (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        help="Run one offline 504x280 test image instead of starting ROS 2",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("depth_trt_test.png"),
        help="Offline test visualization path (default: depth_trt_test.png)",
    )
    return parser.parse_known_args(args)


def _run_offline_test(engine_path, image_path, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read test image: {image_path}")
    model_input, _ = _prepare_image(image)
    engine = TensorRTEngine(engine_path)
    depth = np.asarray(engine.infer(model_input), dtype=np.float32).squeeze()
    expected_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    if depth.shape != expected_shape:
        raise ValueError(f"Unexpected output shape {depth.shape}; expected {expected_shape}")
    finite = depth[np.isfinite(depth)]
    if not finite.size:
        raise RuntimeError("TensorRT output contains no finite values")
    depth_min = float(finite.min())
    depth_max = float(finite.max())
    print(
        f"input={tuple(model_input.shape)} output={tuple(depth.shape)} "
        f"min={depth_min:.8f} max={depth_max:.8f} "
        f"mean={float(finite.mean()):.8f} nonzero={np.count_nonzero(finite)}/{depth.size}"
    )
    if depth_max > depth_min:
        gray = np.clip(
            (depth - depth_min) / (depth_max - depth_min) * 255.0, 0, 255
        ).astype(np.uint8)
    else:
        gray = np.zeros(depth.shape, dtype=np.uint8)
    visualization = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise RuntimeError(f"Unable to save test output: {output_path}")
    np.save(output_path.with_suffix(".npy"), depth.astype(np.float32))
    print(f"saved visualization: {output_path}")
    print(f"saved raw depth: {output_path.with_suffix('.npy')}")


def main(args=None):
    cli_args, ros_args = _parse_arguments(args)
    engine_path = cli_args.engine.expanduser().resolve()
    if cli_args.test_image is not None:
        _run_offline_test(
            engine_path,
            cli_args.test_image.expanduser().resolve(),
            cli_args.test_output.expanduser().resolve(),
        )
        return
    rclpy.init(args=ros_args)
    node = DepthAnythingV2TensorRTNode(
        engine_path=engine_path,
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
