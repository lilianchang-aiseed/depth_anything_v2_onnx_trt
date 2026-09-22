#!/usr/bin/env python3
"""
multi_mipi_camera_raw.py

4x IMX219 MIPI camera node with an internal synchronized publish timer.

GStreamer capture runs at 20 fps per camera (keeping the ISP pipeline warm).
ROS publishing happens ONLY in a single node-level timer callback at
`publish_freq` Hz, giving all 4 cameras the SAME header timestamp.

Default publish_freq is 20 Hz (full GStreamer rate).  The downstream
lidar_snapshot_node buffers these frames and subsamples to LiDAR rate
(≈10 Hz), so this node should publish at least as fast as the LiDAR to
give the sync node enough candidates per window.

Usage:
    ros2 run camarray_ros2 multi_mipi_camera_raw.py \
        --ros-args -p publish_freq:=20.0

Topics published (at publish_freq Hz):
    /camera_N/image_raw   sensor_msgs/Image  (encoding: bgr8)
    /camera_N/camera_info sensor_msgs/CameraInfo

These are consumed by lidar_snapshot_node, which republishes matched
pairs to /synced/camera_N/* for bag recording.
"""

import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,   # IMX219 native resolution
    capture_height=720,
    display_width=420,   # Downscale for manageable frame size
    display_height=280,
    # display_width=1640,   # Downscale for manageable frame size
    # display_height=1232,
    framerate=60,
    flip_method=0,
):
    """
    GStreamer pipeline for Jetson Orin nvarguscamerasrc.

    nvarguscamerasrc routes through the Jetson ISP → 8-bit demosaiced BGR.
    True 10-bit Bayer capture requires bypassing the ISP via nvv4l2src.
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CameraStreamer:
    """
    Capture-only thread for a single camera.

    Runs the GStreamer pipeline at 20 fps and stores the latest frame in a
    thread-safe buffer. Publishing is handled by the node-level timer so that
    all cameras share a single ROS timestamp.

    cv2_to_imgmsg encoding (~3.4 MB/frame) is performed in this capture
    thread so that the timer callback only needs to update the header stamp
    and call publish() — keeping the callback well under the timer period.
    """

    def __init__(self, node: Node, sensor_id: int, bridge: 'CvBridge'):
        self.node = node
        self.sensor_id = sensor_id
        self._bridge = bridge
        self._latest_msg = None        # pre-encoded Image (stamp left at zero)
        self._latest_capture_time = None  # ROS time recorded at cap.read()
        self._lock = threading.Lock()

        pipeline = gstreamer_pipeline(sensor_id=sensor_id)
        self.node.get_logger().info(f"Starting camera {sensor_id}...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.node.get_logger().error(f"Failed to open camera {sensor_id}")
            self.is_running = False
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Continuously grab frames, encode them, and store the latest one."""
        while self.is_running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Timestamp immediately after cap.read() — closest approximation
            # to shutter time available without hardware timestamping.
            capture_time = self.node.get_clock().now().to_msg()
            msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            with self._lock:
                self._latest_msg = msg
                self._latest_capture_time = capture_time

    def get_latest_msg(self):
        """Return (Image msg, capture stamp) for the most recent frame.

        The capture stamp is recorded at cap.read() return time, not at
        publication time, so it reflects when the image was actually taken.
        Returns (None, None) before the first frame arrives.
        """
        with self._lock:
            return self._latest_msg, self._latest_capture_time

    def stop(self):
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()


class MultiMipiCameraRawNode(Node):
    def __init__(self):
        super().__init__('multi_mipi_camera_raw_node')

        # ROS 2 parameter: publishing frequency in Hz.
        # Set this to match your desired recording / lidar-sync frequency.
        self.declare_parameter('publish_freq', 60.0)
        publish_freq = (
            self.get_parameter('publish_freq').get_parameter_value().double_value
        )

        self.bridge = CvBridge()
        self.streamers: list[CameraStreamer] = []
        self._img_pubs = []
        self._info_pubs = []

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        for i in range(4):
            self.streamers.append(CameraStreamer(self, sensor_id=i, bridge=self.bridge))
            self._img_pubs.append(
                self.create_publisher(Image, f'camera_{i}/image_raw', 10)
            )
            self._info_pubs.append(
                self.create_publisher(CameraInfo, f'camera_{i}/camera_info', 10)
            )
            # Allow Argus Daemon to complete CSI handshake before next sensor
            time.sleep(1.0)

        # Single shared timer → all cameras published in one callback with the
        # same timestamp. No separate throttle node required.
        self.create_timer(1.0 / publish_freq, self._publish_cb)
        self.get_logger().info(
            f"Publishing {len(self.streamers)} cameras at {publish_freq} Hz"
        )

    def _publish_cb(self):
        """Publish the latest pre-encoded frame from every camera.

        Encoding happens in each CameraStreamer's capture thread, so this
        callback only stamps and forwards an already-built Image buffer.
        A single timestamp is taken once per callback so that all cameras
        share the same wall-clock trigger time for lidar synchronisation.
        """
        publish_stamp = self.get_clock().now().to_msg()
        for i, streamer in enumerate(self.streamers):
            if not streamer.is_running:
                continue
            img_msg, capture_stamp = streamer.get_latest_msg()
            if img_msg is None:
                continue

            frame_id = f'camera_{i}_optical_frame'

            img_msg.header.stamp = publish_stamp
            img_msg.header.frame_id = frame_id
            self._img_pubs[i].publish(img_msg)

            info_msg = CameraInfo()
            info_msg.header.stamp = publish_stamp
            info_msg.header.frame_id = frame_id
            info_msg.width = img_msg.width
            info_msg.height = img_msg.height
            info_msg.distortion_model = "plumb_bob"
            # NOTE: Replace zeros with D, K, R, P matrices after calibration.
            self._info_pubs[i].publish(info_msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down cameras...")
        for streamer in self.streamers:
            streamer.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiMipiCameraRawNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
