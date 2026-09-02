#!/usr/bin/env python3
"""Record synchronized stereo images/disparity and RealSense depth as NPZ.

Default topics:
  /stereo_1_0/left/image_rect
  /stereo_1_0/right/image_rect
  /stereo_1_0/disparity
  /d455/d455_node/depth/image_rect_raw
  /d435/d435_node/depth/image_rect_raw

Example:
  python3 Depth-Anything-V2/ml_gt_generate/tools/ros_topics_to_npz.py \
    --out-dir Depth-Anything-V2/ml_gt_generate/dataset/train_1_0-5_realsense

Each ``sample_NNNN.npz`` contains:
  left, right, disp, depth_d455, depth_d435,
  stamp_left_ns, stamp_right_ns, stamp_disp_ns,
  stamp_d455_ns, stamp_d435_ns

Each additional ``--add-topic KEY=TOPIC`` stores:
  KEY, stamp_KEY_ns, encoding_KEY

RealSense depth is stored as float32 metres. ``disp`` is the unmodified
float32 stereo disparity in pixels. All timestamps are the original ROS header
timestamps; they allow synchronization quality to be checked later.
This recorder does not spatially project either RealSense depth image into the
stereo-left image frame.
"""

import argparse
import sys
from pathlib import Path

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_ns(msg):
    return np.int64(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)


def image_array(msg):
    formats = {
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "mono8": (np.uint8, 1),
        "8UC1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1),
        "32FC1": (np.float32, 1),
    }
    if msg.encoding not in formats:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    base_dtype, channels = formats[msg.encoding]
    dtype = np.dtype(base_dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = dtype.itemsize
    row_items = msg.step // itemsize
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_items)
    array = array[:, :msg.width * channels]
    if channels > 1:
        array = array.reshape(msg.height, msg.width, channels)
    else:
        array = array.reshape(msg.height, msg.width)
    return np.asarray(array, dtype=base_dtype).copy()


def depth_metres(msg, depth_unit):
    depth = image_array(msg)
    if msg.encoding in ("mono16", "16UC1"):
        saturated = depth == np.iinfo(np.uint16).max
        depth = depth.astype(np.float32) * depth_unit
        depth[saturated] = 0.0
    elif msg.encoding == "32FC1":
        depth = depth.astype(np.float32)
    else:
        raise ValueError(f"depth topic must be 16UC1/mono16/32FC1, got {msg.encoding}")
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def disparity_pixels(msg):
    if msg.encoding != "32FC1":
        raise ValueError(f"disparity topic must be 32FC1, got {msg.encoding}")
    return image_array(msg).astype(np.float32, copy=False)


class NpzRecorder(Node):
    def __init__(self, args):
        super().__init__("stereo_realsense_npz_recorder")
        self.args = args
        self.out_dir = Path(args.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index = args.start_index

        self.extra_topics = []
        for spec in args.add_topic:
            if "=" not in spec:
                raise ValueError(
                    f"invalid --add-topic {spec!r}; expected KEY=/topic/name"
                )
            key, topic = (part.strip() for part in spec.split("=", 1))
            if not key.isidentifier() or not topic.startswith("/"):
                raise ValueError(
                    f"invalid --add-topic {spec!r}; KEY must be an identifier "
                    "and TOPIC must start with /"
                )
            if key in {"left", "right", "disp", "depth_d455", "depth_d435"}:
                raise ValueError(f"reserved NPZ key in --add-topic: {key}")
            if any(existing_key == key for existing_key, _ in self.extra_topics):
                raise ValueError(f"duplicate --add-topic key: {key}")
            self.extra_topics.append((key, topic))

        topics = [
            args.left_topic,
            args.right_topic,
            args.disparity_topic,
            args.d455_topic,
            args.d435_topic,
        ]
        topics.extend(topic for _, topic in self.extra_topics)
        subscribers = [
            message_filters.Subscriber(
                self, Image, topic, qos_profile=qos_profile_sensor_data
            )
            for topic in topics
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=args.queue_size, slop=args.slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.record)
        self.get_logger().info(
            f"Recording synchronized samples to {self.out_dir} "
            f"(slop={args.slop * 1000:.1f} ms)"
        )
        for key, topic in self.extra_topics:
            self.get_logger().info(f"Additional image: {topic} -> NPZ[{key!r}]")

    def record(self, left_msg, right_msg, disp_msg, d455_msg, d435_msg, *extra_msgs):
        try:
            left = image_array(left_msg)
            if left_msg.encoding == "rgb8":
                left = left[..., ::-1].copy()  # Store left consistently as BGR.
            right = image_array(right_msg)
            if right_msg.encoding == "rgb8":
                right = right[..., ::-1].copy()  # Store right consistently as BGR.
            disp = disparity_pixels(disp_msg)
            d455 = depth_metres(d455_msg, self.args.depth_unit)
            d435 = depth_metres(d435_msg, self.args.depth_unit)
            extras = {
                key: image_array(msg)
                for (key, _), msg in zip(self.extra_topics, extra_msgs)
            }
        except (ValueError, TypeError) as error:
            self.get_logger().error(str(error))
            return

        output = self.out_dir / f"sample_{self.index:04d}.npz"
        if output.exists() and not self.args.overwrite:
            self.get_logger().error(f"Refusing to overwrite {output}; stopping")
            rclpy.shutdown()
            return
        values = dict(
            left=left,
            right=right,
            disp=disp,
            depth_d455=d455,
            depth_d435=d435,
            stamp_left_ns=stamp_ns(left_msg),
            stamp_right_ns=stamp_ns(right_msg),
            stamp_disp_ns=stamp_ns(disp_msg),
            stamp_d455_ns=stamp_ns(d455_msg),
            stamp_d435_ns=stamp_ns(d435_msg),
        )
        for (key, _), msg in zip(self.extra_topics, extra_msgs):
            values[key] = extras[key]
            values[f"stamp_{key}_ns"] = stamp_ns(msg)
            values[f"encoding_{key}"] = np.str_(msg.encoding)
        np.savez_compressed(output, **values)

        dt455 = abs(int(stamp_ns(d455_msg) - stamp_ns(left_msg))) / 1e6
        dt435 = abs(int(stamp_ns(d435_msg) - stamp_ns(left_msg))) / 1e6
        dt_right = abs(int(stamp_ns(right_msg) - stamp_ns(left_msg))) / 1e6
        dt_disp = abs(int(stamp_ns(disp_msg) - stamp_ns(left_msg))) / 1e6
        extra_deltas = "".join(
            f", dt_{key}={abs(int(stamp_ns(msg) - stamp_ns(left_msg))) / 1e6:.2f} ms"
            for (key, _), msg in zip(self.extra_topics, extra_msgs)
        )
        self.get_logger().info(
            f"[{self.index:04d}] {output.name} | "
            f"dt_right={dt_right:.2f} ms, dt_disp={dt_disp:.2f} ms, "
            f"dt455={dt455:.2f} ms, dt435={dt435:.2f} ms{extra_deltas}"
        )
        self.index += 1
        if self.args.max_samples and self.index - self.args.start_index >= self.args.max_samples:
            self.get_logger().info(f"Recorded {self.args.max_samples} samples")
            rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--left-topic", default="/stereo_1_0/left/image_rect")
    parser.add_argument("--right-topic", default="/stereo_1_0/right/image_rect")
    parser.add_argument("--disparity-topic", default="/stereo_1_0/disparity")
    parser.add_argument("--d455-topic", default="/d455/d455_node/depth/image_rect_raw")
    parser.add_argument("--d435-topic", default="/d435/d435_node/depth/image_rect_raw")
    parser.add_argument(
        "--add-topic", action="append", default=[], metavar="KEY=/TOPIC",
        help="Add a synchronized sensor_msgs/Image topic to each NPZ; repeatable",
    )
    parser.add_argument("--slop", type=float, default=0.05,
                        help="Approximate sync tolerance in seconds")
    parser.add_argument("--queue-size", type=int, default=50)
    parser.add_argument("--depth-unit", type=float, default=0.001,
                        help="Metres per uint16 depth unit")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Stop after N samples; 0 records until Ctrl-C")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_known_args()


def main():
    args, ros_args = parse_args()
    rclpy.init(args=[sys.argv[0], *ros_args])
    node = NpzRecorder(args)
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
