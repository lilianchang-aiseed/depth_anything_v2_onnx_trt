#!/usr/bin/env python3
"""
lidar_snapshot_node.py — LiDAR-driven camera/LiDAR synchronizer.

The Hesai driver uses an internal clock (seconds since power-on) for its
PointCloud2 header stamps, while cameras use Unix wall time; direct header-stamp
comparison therefore never works.  Instead this node compares:

  • camera frames   – wall-clock time recorded by THIS relay when the message arrives
  • LiDAR frames    – wall-clock time recorded by THIS relay when the message arrives

Both timestamps are taken at the same node with the same clock, so their
difference reflects the true sensor-to-sensor temporal offset.  DDS transport
latency affects both sides equally and cancels out in the delta.  The published
camera messages still carry their original header.stamp from the camera node.

Synchronisation strategy (LiDAR-driven)
───────────────────────────────────────
The camera node publishes all 4 cameras at ~20 Hz (native GStreamer rate).
Each per-camera rolling buffer holds the last CAM_BUF frames.

On every /lidar_points arrival (≈10 Hz):
  1. Record relay receive-time  recv_ns = now().
  2. For each camera buffer find the frame whose header stamp is closest
     to recv_ns.
  3. If ALL cameras have a match within SLOP_NS, publish the matched set
     to /synced/* and clear every buffer so the next sync window is fresh.
  4. If ANY camera has no frame in range, the LiDAR frame is discarded.

Halting caching during publish
──────────────────────────────
The _writing flag is set inside the lock before any publish() call and
cleared after the last publish() returns.  Both _lidar_cb and _cam_cb
check this flag at entry; while True they silently drop incoming data.
This prevents disk-I/O back-pressure from stacking stale frames in the
buffers while a write cycle is in progress.

Published topics (ros2 bag record subscribes to these):
  /synced/camera_N/image_raw      – matched Image for camera N
  /synced/camera_N/camera_info    – matching CameraInfo (width/height only until calibrated)
  /synced/lidar_points            – matched PointCloud2, re-stamped to recv_ns
  /synced/lidar_imu               – most-recent IMU sample at match time
  /synced/lidar_packets_loss      – most-recent LossPacket (requires hesai_ros_driver)
"""

import sys
from collections import deque
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2

try:
    from hesai_ros_driver.msg import LossPacket
    _HAS_LOSS_PACKET = True
except ImportError:
    _HAS_LOSS_PACKET = False

NUM_CAMERAS = 4


class LiDARDrivenSyncNode(Node):
    # Maximum wall-clock delta (ns) between LiDAR recv time and camera recv time.
    # Both are measured at THIS relay node, so DDS latency cancels.
    # Camera at 20 Hz → inter-frame gap 50 ms → worst-case true delta ~25 ms.
    SLOP_NS = 25_000_000   # 25 ms

    # Maximum header.stamp difference (ns) between consecutive camera pairs
    # 0→1, 1→2, 2→3, 3→0.  All four cameras share a single publish_stamp in
    # multi_mipi_camera_raw_node, so under normal operation this spread is ~0.
    # A value above this threshold means the matched frames came from different
    # timer ticks and should be rejected.
    CAM_SPREAD_NS = 25_000_000   # 25 ms

    # Rolling buffer depth per camera.  At 20 Hz, 10 frames = 500 ms of
    # history — well beyond the post-write halt period at typical freq values.
    CAM_BUF = 2

    def __init__(self) -> None:
        super().__init__('lidar_driven_sync_node')

        # Output rate: after each successful write, halt all buffering for
        # 1/freq seconds so that the bag is written at ~freq Hz regardless
        # of the native LiDAR rate.
        self.declare_parameter('freq', 0.5)
        self.declare_parameter('lidar_points_topic', '/lidar_points')
        self.declare_parameter('lidar_imu_topic', '/lidar_imu')
        self.declare_parameter('lidar_loss_topic', '/lidar_packets_loss')
        self.declare_parameter('enable_loss_packet', True)
        freq = self.get_parameter('freq').get_parameter_value().double_value
        self._lidar_points_topic = (
            self.get_parameter('lidar_points_topic').value
        )
        self._lidar_imu_topic = self.get_parameter('lidar_imu_topic').value
        self._lidar_loss_topic = self.get_parameter('lidar_loss_topic').value
        self._enable_loss_packet = self.get_parameter('enable_loss_packet').value
        self._period_ns: int = int(1.0 / freq * 1_000_000_000)

        self._lock = Lock()
        self._writing: bool = False
        # Wall-clock ns before which all callbacks drop incoming data.
        # Set to now() + _period_ns immediately after each successful write.
        self._resume_time: int = 0

        # Per-camera rolling buffer: deque of (recv_ns, Image)
        self._cam_bufs: list[deque] = [
            deque(maxlen=self.CAM_BUF) for _ in range(NUM_CAMERAS)
        ]

        self._latest_imu: Imu | None = None
        self._latest_imu_recv_ns: int = 0
        self._latest_loss = None

        # Diagnostics counters
        self._cnt_lidar      = 0
        self._cnt_written    = 0
        self._cnt_miss_empty  = 0   # LiDAR arrived before any camera data
        self._cnt_miss_slop   = 0   # nearest camera frame outside SLOP
        self._cnt_miss_spread = 0   # consecutive camera header.stamp delta > CAM_SPREAD_NS
        self._cnt_halt        = 0   # frame dropped during post-write halt period
        self._cnt_skip        = 0   # frame dropped during publish lock

        # ── QoS profiles ──────────────────────────────────────────────────
        be_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        rel_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # RELIABLE so ros2 bag record never drops a synced frame
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_img:  list = []
        self._pub_info: list = []
        for k in range(NUM_CAMERAS):
            self._pub_img.append(
                self.create_publisher(Image,      f'/synced/camera_{k}/image_raw',   pub_qos)
            )
            self._pub_info.append(
                self.create_publisher(CameraInfo, f'/synced/camera_{k}/camera_info', pub_qos)
            )

        self._pub_points = self.create_publisher(PointCloud2, '/synced/lidar_points',       pub_qos)
        self._pub_imu    = self.create_publisher(Imu,         '/synced/lidar_imu',           pub_qos)
        if _HAS_LOSS_PACKET and self._enable_loss_packet:
            self._pub_loss = self.create_publisher(
                LossPacket, '/synced/lidar_packets_loss', pub_qos
            )
        elif self._enable_loss_packet:
            self._pub_loss = None
            self.get_logger().warn(
                'hesai_ros_driver not in sourced workspace; '
                '/synced/lidar_packets_loss will NOT be published.'
            )
        else:
            self._pub_loss = None

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            PointCloud2, self._lidar_points_topic, self._lidar_cb, be_qos
        )
        self.create_subscription(
            Imu, self._lidar_imu_topic, self._imu_cb, be_qos
        )
        if _HAS_LOSS_PACKET and self._enable_loss_packet:
            self.create_subscription(
                LossPacket, self._lidar_loss_topic, self._loss_cb, be_qos
            )

        for k in range(NUM_CAMERAS):
            self.create_subscription(
                Image,
                f'/camera_{k}/image_raw',
                self._make_cam_cb(k),
                rel_qos,
            )

        self.create_timer(5.0, self._log_diagnostics)
        self.get_logger().info(
            f'LiDAR-driven sync started  '
            f'freq={freq} Hz  '
            f'period={self._period_ns // 1_000_000} ms  '
            f'slop={self.SLOP_NS // 1_000_000} ms  '
            f'cam_buf={self.CAM_BUF} frames/camera'
            f'  points={self._lidar_points_topic}'
            f'  imu={self._lidar_imu_topic}'
        )

    # ─────────────────────────────────────────────────────────────────────
    # Subscription callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _make_cam_cb(self, cam_id: int):
        """Return a closure that buffers frames for camera *cam_id*.

        recv_ns is taken at THIS relay node — the same clock and measurement
        point as the lidar recv_ns in _lidar_cb — so DDS latency cancels in
        the delta and the SLOP check reflects true sensor temporal offset.
        """
        def _cb(msg: Image) -> None:
            recv_ns = self.get_clock().now().nanoseconds
            with self._lock:
                if recv_ns < self._resume_time:
                    self._cnt_halt += 1
                    return
                if self._writing:
                    self._cnt_skip += 1
                    return
                self._cam_bufs[cam_id].append((recv_ns, msg))
        return _cb

    def _imu_cb(self, msg: Imu) -> None:
        recv_ns = self.get_clock().now().nanoseconds
        with self._lock:
            self._latest_imu        = msg
            self._latest_imu_recv_ns = recv_ns

    def _loss_cb(self, msg) -> None:
        with self._lock:
            self._latest_loss = msg

    def _lidar_cb(self, msg: PointCloud2) -> None:
        recv_ns = self.get_clock().now().nanoseconds

        with self._lock:
            if recv_ns < self._resume_time:
                self._cnt_halt += 1
                return
            if self._writing:
                self._cnt_skip += 1
                return

            self._cnt_lidar += 1

            # ── Step 1: ensure every camera has at least one buffered frame ──
            for k, buf in enumerate(self._cam_bufs):
                if not buf:
                    self._cnt_miss_empty += 1
                    return  # no data yet for camera k

            # ── Step 2: find nearest camera frame per camera ──────────────
            matched: list[tuple[int, Image]] = []   # (cam_recv_ns, Image) per camera
            for k, buf in enumerate(self._cam_bufs):
                best_stamp_ns, best_img = min(buf, key=lambda t: abs(t[0] - recv_ns))
                delta_ns = abs(best_stamp_ns - recv_ns)
                if delta_ns > self.SLOP_NS:
                    self._cnt_miss_slop += 1
                    self.get_logger().debug(
                        f'cam_{k} best delta={delta_ns // 1_000_000} ms '
                        f'> slop={self.SLOP_NS // 1_000_000} ms — discarding LiDAR frame'
                    )
                    return
                matched.append((best_stamp_ns, best_img))

            # ── Step 2b: consecutive-camera header.stamp spread check ─────
            # Pairs: 0→1, 1→2, 2→3, 3→0.
            # All cameras share publish_stamp from one timer tick, so any
            # large spread means different ticks were matched across cameras.
            stamps = [
                img.header.stamp.sec * 1_000_000_000 + img.header.stamp.nanosec
                for _, img in matched
            ]
            n = len(stamps)
            for i in range(n):
                diff = abs(stamps[i] - stamps[(i + 1) % n])
                if diff > self.CAM_SPREAD_NS:
                    self._cnt_miss_spread += 1
                    self.get_logger().debug(
                        f'cam_{i}→cam_{(i + 1) % n} header.stamp diff='
                        f'{diff // 1_000_000} ms '
                        f'> spread={self.CAM_SPREAD_NS // 1_000_000} ms '
                        f'— discarding LiDAR frame'
                    )
                    return

            # ── Step 3: all cameras matched — halt ingestion and publish ──
            self._writing = True

            # Snapshot ancillary data under the lock before releasing
            imu_snap     = self._latest_imu
            imu_recv_ns  = self._latest_imu_recv_ns
            loss_snap    = self._latest_loss

            # Clear all buffers so the next window starts fresh.
            # New frames will accumulate after _writing is cleared.
            for buf in self._cam_bufs:
                buf.clear()

        # ── Publish outside the lock (DDS serialisation is I/O) ──────────
        #
        # Re-stamp the LiDAR message with the relay receive-time so all
        # /synced/* topics share a common wall-clock reference.
        lidar_stamp_sec  = recv_ns // 1_000_000_000
        lidar_stamp_nsec = recv_ns %  1_000_000_000

        msg.header.stamp.sec    = lidar_stamp_sec
        msg.header.stamp.nanosec = lidar_stamp_nsec
        self._pub_points.publish(msg)

        for k, (_, img_msg) in enumerate(matched):
            frame_id = f'camera_{k}_optical_frame'

            img_msg.header.frame_id = frame_id
            self._pub_img[k].publish(img_msg)

            info = CameraInfo()
            info.header.stamp    = img_msg.header.stamp   # keep original camera stamp
            info.header.frame_id = frame_id
            info.width           = img_msg.width
            info.height          = img_msg.height
            info.distortion_model = 'plumb_bob'
            # NOTE: populate D, K, R, P after sensor calibration.
            self._pub_info[k].publish(info)

        if imu_snap is not None:
            imu_snap.header.stamp.sec    = imu_recv_ns // 1_000_000_000
            imu_snap.header.stamp.nanosec = imu_recv_ns %  1_000_000_000
            self._pub_imu.publish(imu_snap)

        if loss_snap is not None and self._pub_loss is not None:
            self._pub_loss.publish(loss_snap)

        with self._lock:
            self._cnt_written += 1
            self._writing = False
            # Block new data for one full period so the effective output rate
            # matches freq Hz, independent of the native LiDAR rate.
            self._resume_time = self.get_clock().now().nanoseconds + self._period_ns

        self.get_logger().debug(
            f'Published synced frame  lidar_recv_ns={recv_ns}  '
            + '  '.join(
                f'cam{k}Δ={abs(s - recv_ns) // 1_000_000}ms'
                for k, (s, _) in enumerate(matched)
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    def _log_diagnostics(self) -> None:
        with self._lock:
            cam_depths = [len(b) for b in self._cam_bufs]
        self.get_logger().info(
            f'[diag] lidar_rx={self._cnt_lidar}  '
            f'written={self._cnt_written}  '
            f'miss_empty={self._cnt_miss_empty}  '
            f'miss_slop={self._cnt_miss_slop}  '
            f'miss_spread={self._cnt_miss_spread}  '
            f'halt={self._cnt_halt}  '
            f'skip_write={self._cnt_skip}  '
            f'cam_buf_depths={cam_depths}'
        )
        if self._cnt_lidar == 0:
            self.get_logger().warn(
                f'No {self._lidar_points_topic} received — '
                'is the selected LiDAR driver running, and is its point cloud '
                'type sensor_msgs/msg/PointCloud2?'
            )
        elif self._cnt_written == 0:
            self.get_logger().warn(
                f'LiDAR arriving but no frame written — '
                f'empty={self._cnt_miss_empty}  slop={self._cnt_miss_slop}  '
                f'skip={self._cnt_skip}  '
                f'(check camera publish rate and SLOP_NS={self.SLOP_NS // 1_000_000} ms)'
            )


# ═════════════════════════════════════════════════════════════════════════
def main() -> None:
    rclpy.init(args=sys.argv)
    node = LiDARDrivenSyncNode()
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
