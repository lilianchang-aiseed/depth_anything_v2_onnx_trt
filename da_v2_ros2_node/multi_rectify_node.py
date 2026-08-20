#!/usr/bin/env python3
"""Rectify four fixed stereo pairs without running a stereo depth model."""

import argparse
import collections
import inspect
import itertools
from pathlib import Path
import queue
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
import yaml


OUTPUT_SIZE = 320
HALF_ANGLE = np.pi / 4
STEREO_PAIRS = ((0, 3), (1, 0), (2, 1), (3, 2))


def _find_rectify_dir(requested):
    if requested:
        candidates = [Path(requested).expanduser()]
    else:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "rectify",
            here.parent / "rectify",
            here.parent.parent / "rectify",
            Path("/home/nvidia/ros_stereo/rectify"),
        ]
    for candidate in candidates:
        if (candidate / "distortion_models.py").is_file() and (
            candidate / "rectify_utils.py"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find rectify directory containing distortion_models.py and "
        "rectify_utils.py; pass --rectify-dir"
    )


def _find_calibration(rectify_dir, requested):
    if requested:
        candidates = [Path(requested).expanduser()]
    else:
        candidates = [
            rectify_dir / "stereo_calib_ds-camchain_from_json.yaml",
            rectify_dir / "stereo_calib_ds-camchain.yaml",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find multi-pair calibration; pass --calibration")


def _image_msg_to_bgr(message):
    conversions = {"bgr8": (3, None), "rgb8": (3, "rgb"), "mono8": (1, "gray")}
    encoding = message.encoding.lower()
    if encoding not in conversions:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    channels, conversion = conversions[encoding]
    row_bytes = int(message.width) * channels
    required = int(message.step) * int(message.height)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size < required or message.step < row_bytes:
        raise ValueError("Invalid ROS image buffer or step")
    pixels = data[:required].reshape(message.height, message.step)[:, :row_bytes]
    if channels == 1:
        image = pixels.reshape(message.height, message.width)
        image = np.repeat(image[:, :, None], 3, axis=2)
    else:
        image = pixels.reshape(message.height, message.width, channels)
        if conversion == "rgb":
            image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _ds_intrinsics(camera, scale=1.0):
    values = camera["intrinsics"]
    return (
        values[0], values[1], scale * values[2], scale * values[3],
        scale * values[4], scale * values[5],
    )


class MultiRectifyNode(Node):
    def __init__(self, calibration, rectify_dir):
        super().__init__("multi_stereo_rectify")
        sys.path.insert(0, str(rectify_dir))
        from distortion_models import unproject_double_sphere_pixels
        from rectify_utils import (
            epiploar_planes_from_extrinsics,
            rasterize_points_map,
            rasterize_points_to_image,
            shape_to_pixel_grid,
        )

        self._unproject = unproject_double_sphere_pixels
        self._epipolar_planes = epiploar_planes_from_extrinsics
        self._rasterize_map = rasterize_points_map
        self._rasterize_map_uses_b2n = (
            "b2n" in inspect.signature(rasterize_points_map).parameters
        )
        self._rasterize_image = rasterize_points_to_image
        self._pixel_grid = shape_to_pixel_grid

        with calibration.open("r", encoding="utf-8") as stream:
            all_calibration = yaml.safe_load(stream)
        self._pair_calibration = {}
        for left, right in STEREO_PAIRS:
            key = f"cam_pair_{left}_{right}"
            if key not in all_calibration:
                raise KeyError(f"Missing {key} in {calibration}")
            self._pair_calibration[(left, right)] = all_calibration[key]

        self.get_logger().info(f"Calibration: {calibration}")
        self._rect_maps = {
            pair: self._build_rect_maps(config)
            for pair, config in self._pair_calibration.items()
        }

        self._rect_publishers = {}
        for left, right in STEREO_PAIRS:
            for side in ("left", "right"):
                topic = f"/stereo_{left}_{right}/{side}/image_rect"
                self._rect_publishers[(left, right, side)] = self.create_publisher(
                    Image, topic, 1
                )
                self.get_logger().info(f"Publishing {topic}")

        self._camera_indices = (0, 1, 2, 3)
        self._pools = {index: collections.deque(maxlen=2) for index in self._camera_indices}
        self._pool_lock = threading.Lock()
        self._last_dispatch_ns = 0
        self._work_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        qos = QoSProfile(depth=4)
        self._camera_subscriptions = []
        for index in self._camera_indices:
            subscription = self.create_subscription(
                Image,
                f"/camera_{index}/image_raw",
                lambda message, camera=index: self._camera_callback(camera, message),
                qos,
            )
            self._camera_subscriptions.append(subscription)
        self.get_logger().info("Ready: four raw cameras -> eight rectified images")

    def _build_rect_maps(self, calibration):
        camera0 = calibration["cam0"]
        camera1 = calibration["cam1"]
        calibration_width, calibration_height = camera0["resolution"]
        width, height = calibration.get(
            "stream_resolution", (calibration_width, calibration_height)
        )
        scale = height / calibration_height
        pixels = self._pixel_grid((height, width), stride=1)
        rays0, _ = self._unproject(pixels, *_ds_intrinsics(camera0, scale))
        rays1, _ = self._unproject(pixels, *_ds_intrinsics(camera1, scale))

        if "R" in camera1 and "t" in camera1:
            rotation_matrix = np.asarray(camera1["R"], dtype=np.float32)
            translation_vector = np.asarray(camera1["t"], dtype=np.float32)
            rotation = rotation_matrix.T
            translation = rotation_matrix @ translation_vector
        else:
            transform = np.asarray(camera1["T_cn_cnm1"], dtype=np.float32)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]

        normal0, normal1 = self._epipolar_planes(rotation, translation)
        angle_argument = (
            {"b2n": HALF_ANGLE}
            if self._rasterize_map_uses_b2n
            else {"eps": HALF_ANGLE}
        )
        map0, coordinates = self._rasterize_map(
            normal0.astype(np.float32), rays0.astype(np.float32), translation,
            n=OUTPUT_SIZE, **angle_argument,
        )
        camera1_coordinates = [rotation.T @ axis for axis in coordinates]
        map1, _ = self._rasterize_map(
            normal1.astype(np.float32), rays1.astype(np.float32), translation,
            n=OUTPUT_SIZE, coord=camera1_coordinates, **angle_argument,
        )
        return map0, map1

    def _camera_callback(self, camera, message):
        try:
            image = _image_msg_to_bgr(message)
        except Exception as error:
            self.get_logger().error(f"camera_{camera} decode failed: {error}")
            return
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        with self._pool_lock:
            self._pools[camera].append((stamp_ns, image, message.header))
            work = self._assemble_latest_set()
        if work is None:
            return
        try:
            self._work_queue.put_nowait(work)
        except queue.Full:
            try:
                self._work_queue.get_nowait()
                self._work_queue.task_done()
            except queue.Empty:
                pass
            self._work_queue.put_nowait(work)

    def _assemble_latest_set(self):
        pools = [list(self._pools[index]) for index in self._camera_indices]
        if any(not pool for pool in pools):
            return None
        combination = min(
            itertools.product(*pools),
            key=lambda items: max(item[0] for item in items) - min(item[0] for item in items),
        )
        median_ns = sorted(item[0] for item in combination)[len(combination) // 2]
        if median_ns <= self._last_dispatch_ns:
            return None
        self._last_dispatch_ns = median_ns
        return {
            index: (combination[position][1], combination[position][2])
            for position, index in enumerate(self._camera_indices)
        }

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                camera_set = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._rectify_and_publish(camera_set)
            except Exception as error:
                self.get_logger().error(f"Rectification failed: {error}")
            finally:
                self._work_queue.task_done()

    def _rectify_and_publish(self, camera_set):
        for left, right in STEREO_PAIRS:
            map_left, map_right = self._rect_maps[(left, right)]
            left_image, left_header = camera_set[left]
            right_image, right_header = camera_set[right]
            rect_left, _ = self._rasterize_image(
                *map_left, left_image, n=OUTPUT_SIZE, output_type="rgb"
            )
            rect_right, _ = self._rasterize_image(
                *map_right, right_image, n=OUTPUT_SIZE, output_type="rgb"
            )
            self._publish(rect_left, left_header, self._rect_publishers[(left, right, "left")])
            self._publish(rect_right, right_header, self._rect_publishers[(left, right, "right")])

    @staticmethod
    def _publish(image, header, publisher):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        message = Image()
        message.header = header
        message.height, message.width = image.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = image.strides[0]
        message.data = image.tobytes()
        publisher.publish(message)

    def destroy_node(self):
        self._stop_event.set()
        if hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="Four-pair rectification-only ROS 2 node")
    parser.add_argument("--rectify-dir")
    parser.add_argument("--calibration")
    parsed, ros_args = parser.parse_known_args(args)
    rectify_dir = _find_rectify_dir(parsed.rectify_dir)
    calibration = _find_calibration(rectify_dir, parsed.calibration)
    rclpy.init(args=ros_args)
    node = MultiRectifyNode(calibration, rectify_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
