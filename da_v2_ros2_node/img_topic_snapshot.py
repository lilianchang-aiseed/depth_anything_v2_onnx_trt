#!/usr/bin/env python3
"""Save one ROS 2 Image message as raw NPY and displayable JPG.

Example:
    python3 img_topic_snapshot.py \
        --img-topic /stereo_0_3/left/relative_depth

The output directory defaults to the directory containing this script.
"""

import argparse
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


SCRIPT_DIR = Path(__file__).resolve().parent

# encoding: (NumPy dtype, channels, conversion to OpenCV BGR)
ENCODINGS = {
    "bgr8": (np.dtype("u1"), 3, None),
    "rgb8": (np.dtype("u1"), 3, cv2.COLOR_RGB2BGR),
    "bgra8": (np.dtype("u1"), 4, cv2.COLOR_BGRA2BGR),
    "rgba8": (np.dtype("u1"), 4, cv2.COLOR_RGBA2BGR),
    "mono8": (np.dtype("u1"), 1, None),
    "gray8": (np.dtype("u1"), 1, None),
    "8uc1": (np.dtype("u1"), 1, None),
    "8uc3": (np.dtype("u1"), 3, None),
    "mono16": (np.dtype("u2"), 1, None),
    "16uc1": (np.dtype("u2"), 1, None),
    "16sc1": (np.dtype("i2"), 1, None),
    "32fc1": (np.dtype("f4"), 1, None),
    "32fc3": (np.dtype("f4"), 3, None),
    "64fc1": (np.dtype("f8"), 1, None),
}


def decode_image(message):
    encoding = message.encoding.lower()
    if encoding not in ENCODINGS:
        raise ValueError(
            f"Unsupported encoding {message.encoding!r}; supported: "
            f"{', '.join(sorted(ENCODINGS))}"
        )

    base_dtype, channels, color_conversion = ENCODINGS[encoding]
    if base_dtype.itemsize > 1:
        byte_order = ">" if message.is_bigendian else "<"
        dtype = base_dtype.newbyteorder(byte_order)
    else:
        dtype = base_dtype

    if message.step % dtype.itemsize:
        raise ValueError(
            f"Image step {message.step} is not divisible by {dtype.itemsize}"
        )
    row_elements = message.step // dtype.itemsize
    needed_elements = row_elements * message.height
    buffer = np.frombuffer(message.data, dtype=dtype)
    if buffer.size < needed_elements:
        raise ValueError(
            f"Image contains {buffer.size} elements; expected {needed_elements}"
        )

    pixel_elements = message.width * channels
    if row_elements < pixel_elements:
        raise ValueError(
            f"Image step contains {row_elements} elements per row; "
            f"expected at least {pixel_elements}"
        )
    pixels = buffer[:needed_elements].reshape(message.height, row_elements)
    pixels = pixels[:, :pixel_elements]
    if channels == 1:
        image = pixels.reshape(message.height, message.width)
    else:
        image = pixels.reshape(message.height, message.width, channels)

    # Convert non-native-endian arrays before saving/processing.
    image = np.asarray(image, dtype=base_dtype).copy()
    display_image = image
    if color_conversion is not None:
        display_image = cv2.cvtColor(image, color_conversion)
    return image, display_image


def normalize_for_jpg(image):
    """Convert integer/float mono or multi-channel data to displayable uint8."""
    if image.dtype == np.uint8:
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    values = image.astype(np.float32)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        raise ValueError("Image contains no finite values")

    minimum = float(finite.min())
    maximum = float(finite.max())
    if maximum > minimum:
        normalized = (values - minimum) / (maximum - minimum)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros(values.shape, dtype=np.uint8)

    if normalized.ndim == 2:
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    if normalized.shape[2] == 1:
        return cv2.applyColorMap(normalized[:, :, 0], cv2.COLORMAP_TURBO)
    if normalized.shape[2] == 4:
        return cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)
    return normalized


def safe_topic_name(topic):
    return re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_") or "image"


class ImageSaver(Node):
    def __init__(self, topic, output_dir):
        super().__init__("img_topic_snapshot")
        self._topic = topic
        self._output_dir = output_dir
        self._saved = False
        self._subscription = self.create_subscription(
            Image, topic, self._callback, 1
        )
        self.get_logger().info(f"Waiting for one image on {topic}")

    def _callback(self, message):
        if self._saved:
            return
        try:
            raw, display = decode_image(message)
            jpg = normalize_for_jpg(display)
            stamp = f"{message.header.stamp.sec}_{message.header.stamp.nanosec:09d}"
            base = f"{safe_topic_name(self._topic)}_{stamp}"
            npy_path = self._output_dir / f"{base}.npy"
            jpg_path = self._output_dir / f"{base}.jpg"
            self._output_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, raw)
            if not cv2.imwrite(str(jpg_path), jpg):
                raise RuntimeError(f"OpenCV failed to save {jpg_path}")

            finite = raw[np.isfinite(raw)] if np.issubdtype(raw.dtype, np.number) else raw
            statistics = ""
            if finite.size:
                statistics = (
                    f", min={finite.min()}, max={finite.max()}, "
                    f"mean={float(finite.mean()):.6f}, "
                    f"nonzero={np.count_nonzero(finite)}/{finite.size}"
                )
            print(
                f"encoding={message.encoding}, shape={raw.shape}, dtype={raw.dtype}"
                f"{statistics}\nSaved raw: {npy_path}\nSaved JPG: {jpg_path}"
            )
            self._saved = True
            rclpy.shutdown()
        except Exception as error:
            self.get_logger().error(f"Unable to save image: {error}")
            self._saved = True
            rclpy.shutdown()


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Save one ROS 2 Image topic message as NPY and JPG"
    )
    parser.add_argument("--img-topic", required=True, help="sensor_msgs/Image topic")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help=f"Output directory (default: {SCRIPT_DIR})",
    )
    return parser.parse_known_args(args)


def main(args=None):
    parsed, ros_args = parse_arguments(args)
    rclpy.init(args=ros_args)
    node = ImageSaver(parsed.img_topic, parsed.out_dir.expanduser().resolve())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
