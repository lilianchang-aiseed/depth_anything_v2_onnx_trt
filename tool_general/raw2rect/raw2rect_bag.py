#!/usr/bin/env python3
"""Add calibrated rectified camera topics to a ROS 2 bag offline.

The input bag is read directly; ros2 bag play and a running ROS graph are not
required. Existing configured image_rect topics are preserved by default and
can be replaced explicitly with --replace-old-rect.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

try:
    from .ds_rectifier import load_rectifiers
except ImportError:  # Direct execution: python3 raw2rect_bag.py ...
    from ds_rectifier import load_rectifiers


IMAGE_TYPE = "sensor_msgs/msg/Image"


def decode_image(message) -> np.ndarray:
    """Decode an uncompressed 8-bit sensor_msgs/Image into BGR."""
    encoding = message.encoding.lower()
    conversions = {
        "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "bgra8": (4, cv2.COLOR_BGRA2BGR),
        "rgba8": (4, cv2.COLOR_RGBA2BGR),
        "mono8": (1, cv2.COLOR_GRAY2BGR),
        "8uc1": (1, cv2.COLOR_GRAY2BGR),
    }
    if encoding not in conversions:
        raise ValueError(
            f"Unsupported encoding {message.encoding!r}; expected one of "
            f"{sorted(conversions)}"
        )
    channels, conversion = conversions[encoding]
    row_bytes = int(message.width) * channels
    required = int(message.step) * int(message.height)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if int(message.step) < row_bytes or data.size < required:
        raise ValueError(
            f"Invalid image buffer: {message.width}x{message.height}, "
            f"step={message.step}, bytes={data.size}"
        )
    rows = data[:required].reshape(int(message.height), int(message.step))
    pixels = rows[:, :row_bytes]
    if channels == 1:
        image = pixels.reshape(int(message.height), int(message.width))
    else:
        image = pixels.reshape(int(message.height), int(message.width), channels)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    elif channels == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def input_paths(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input MCAP file or ROS 2 bag directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="New output ROS 2 bag directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "raw2rect_config.yaml",
        help="Rectification configuration YAML",
    )
    parser.add_argument(
        "--max-bag-messages",
        type=int,
        default=None,
        help="Stop after this many input bag messages (smoke testing only)",
    )
    parser.add_argument(
        "--replace-old-rect",
        action="store_true",
        help=(
            "Replace configured image_rect topics already present in the input bag. "
            "Without this flag, existing rect topics are copied unchanged and only "
            "missing rect topics are generated."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_bag_messages is not None and args.max_bag_messages <= 0:
        raise SystemExit("--max-bag-messages must be positive")

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"Output already exists: {out_dir}. Choose a new directory; "
            "this tool never overwrites bags."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    specs, rectifiers, resolved_config = load_rectifiers(args.config)
    input_topic_to_camera = {spec.input_topic: camera_id for camera_id, spec in specs.items()}
    output_topics = {spec.output_topic for spec in specs.values()}
    if len(input_topic_to_camera) != len(specs) or len(output_topics) != len(specs):
        raise ValueError("Configured input and output topics must be unique")

    try:
        from rosbags.highlevel import AnyReader
        from rosbags.rosbag2 import StoragePlugin, Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError("rosbags is required: pip install rosbags") from exc

    source_paths = input_paths(args.input)
    default_typestore = get_typestore(Stores.ROS2_HUMBLE)
    counts: Counter[str] = Counter()
    skipped_existing: Counter[str] = Counter()

    with AnyReader(source_paths, default_typestore=default_typestore) as reader:
        by_topic = {connection.topic: connection for connection in reader.connections}
        missing = [topic for topic in input_topic_to_camera if topic not in by_topic]
        if missing:
            available_camera_topics = sorted(
                connection.topic for connection in reader.connections if "camera" in connection.topic
            )
            raise ValueError(
                f"Missing configured raw topics: {missing}. "
                f"Available camera topics: {available_camera_topics}"
            )
        wrong_type = [
            topic for topic in input_topic_to_camera if by_topic[topic].msgtype != IMAGE_TYPE
        ]
        if wrong_type:
            raise TypeError(f"Configured raw topics are not {IMAGE_TYPE}: {wrong_type}")

        existing_output_topics = output_topics.intersection(by_topic)
        generated_camera_ids = {
            camera_id
            for camera_id, spec in specs.items()
            if args.replace_old_rect or spec.output_topic not in existing_output_topics
        }
        replaced_output_topics = {
            specs[camera_id].output_topic for camera_id in generated_camera_ids
        }.intersection(existing_output_topics)

        # ROS 2 Humble expects rosbag metadata version 8, where
        # offered_qos_profiles is stored as a YAML string. Version 9 writes a
        # structured list which Humble's yaml-cpp parser rejects.
        with Writer(out_dir, version=8, storage_plugin=StoragePlugin.MCAP) as writer:
            copied_connections = {}
            for connection in reader.connections:
                if connection.topic in replaced_output_topics:
                    continue
                copied_connections[connection.id] = writer.add_connection(
                    connection.topic,
                    connection.msgtype,
                    typestore=reader.typestore,
                    serialization_format=connection.ext.serialization_format,
                    offered_qos_profiles=connection.ext.offered_qos_profiles,
                )

            rect_connections = {}
            for camera_id in sorted(generated_camera_ids):
                spec = specs[camera_id]
                source_connection = by_topic[spec.input_topic]
                rect_connections[camera_id] = writer.add_connection(
                    spec.output_topic,
                    IMAGE_TYPE,
                    typestore=reader.typestore,
                    serialization_format=source_connection.ext.serialization_format,
                    offered_qos_profiles=source_connection.ext.offered_qos_profiles,
                )

            processed_messages = 0
            for connection, timestamp, rawdata in reader.messages():
                if args.max_bag_messages is not None and processed_messages >= args.max_bag_messages:
                    break
                processed_messages += 1

                if connection.topic in replaced_output_topics:
                    skipped_existing[connection.topic] += 1
                    continue

                writer.write(copied_connections[connection.id], timestamp, rawdata)
                counts[f"copied:{connection.topic}"] += 1

                camera_id = input_topic_to_camera.get(connection.topic)
                if camera_id is None or camera_id not in generated_camera_ids:
                    continue
                message = reader.deserialize(rawdata, connection.msgtype)
                bgr = decode_image(message)
                rectified = rectifiers[camera_id].rectify(bgr)
                rectified = np.ascontiguousarray(rectified, dtype=np.uint8)
                rect_message = type(message)(
                    header=message.header,
                    height=rectified.shape[0],
                    width=rectified.shape[1],
                    encoding="bgr8",
                    is_bigendian=0,
                    step=rectified.shape[1] * 3,
                    data=rectified.reshape(-1),
                )
                serialized = reader.typestore.serialize_cdr(rect_message, IMAGE_TYPE)
                writer.write(rect_connections[camera_id], timestamp, serialized)
                counts[f"rectified:{specs[camera_id].output_topic}"] += 1

    resolved_config["input"] = str(args.input.expanduser().resolve())
    resolved_config["out_dir"] = str(out_dir)
    resolved_config["replace_existing_rect_topics"] = bool(args.replace_old_rect)
    resolved_config["preserved_existing_rect_topics"] = sorted(
        existing_output_topics - replaced_output_topics
    )
    resolved_config["generated_rect_topics"] = sorted(
        specs[camera_id].output_topic for camera_id in generated_camera_ids
    )
    resolved_config["max_bag_messages"] = args.max_bag_messages
    with (out_dir / "raw2rect_resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False)

    print(f"Done: {out_dir}")
    for camera_id, spec in specs.items():
        if camera_id in generated_camera_ids:
            print(
                f"  generated {spec.input_topic} -> {spec.output_topic}: "
                f"{counts[f'rectified:{spec.output_topic}']}"
            )
        else:
            print(f"  preserved existing {spec.output_topic}")
    for topic, count in sorted(skipped_existing.items()):
        print(f"  replaced existing {topic}: skipped {count} old messages")
    print(f"  processed input bag messages: {processed_messages}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
