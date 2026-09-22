# Depth Anything V2 Fisheye ROS 2 Node

`depth_anything_v2_trt_node-v3.1.py` runs one static batch-4 TensorRT engine
for four AR0234 fisheye cameras. It rectifies each raw image to the left view of
its configured stereo pair, publishes the rectified image independently of
DA-V2 inference, publishes one depth image per camera, and combines the four
metric-depth images into one colored point cloud.

## Topics

| Direction | Topic | Type | Encoding / fields | Size / frame |
|---|---|---|---|---|
| Input | `/camera_0/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | calibration resolution |
| Input | `/camera_1/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | calibration resolution |
| Input | `/camera_2/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | calibration resolution |
| Input | `/camera_3/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | calibration resolution |
| Output | `/camera_0/image_rect` | `sensor_msgs/msg/Image` | `bgr8` | 322x322 |
| Output | `/camera_1/image_rect` | `sensor_msgs/msg/Image` | `bgr8` | 322x322 |
| Output | `/camera_2/image_rect` | `sensor_msgs/msg/Image` | `bgr8` | 322x322 |
| Output | `/camera_3/image_rect` | `sensor_msgs/msg/Image` | `bgr8` | 322x322 |
| Output | `/camera_0/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 322x322 |
| Output | `/camera_1/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 322x322 |
| Output | `/camera_2/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 322x322 |
| Output | `/camera_3/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 322x322 |
| Output | `/depth_anything/point_cloud` | `sensor_msgs/msg/PointCloud2` | `x,y,z,rgb` | `base_link_frd` |

`relative_depth` is the historical topic name. The v3.1 point-cloud path
assumes the loaded TensorRT engine outputs **metric depth in metres**. Do not
make a metric point cloud from an engine that still outputs unscaled relative
depth.

## Rectified images

The node reads:

```text
calib/2026-09-16/stereo_calib_ds-camchain_ar0234.yaml
```

Each camera uses the left rectification map of one pair:

```text
camera 0 -> cam_pair_0_3
camera 1 -> cam_pair_1_0
camera 2 -> cam_pair_2_1
camera 3 -> cam_pair_3_2
```

The Double-Sphere inverse map generates a 322x322, 90-degree circular FOV.
Pixels outside the valid viewing cone are black. The output message copies the
raw image header and timestamp.

When CuPy and all four rectification maps are available, a dedicated
latest-frame-only worker publishes `image_rect` without waiting for DA-V2
inference. The startup log reports:

```text
Fast rect publish ON: image_rect is independent of DA-V2 inference
```

If the fast path is unavailable, rectified images fall back to the inference
worker and therefore publish at the inference rate.

## Point cloud

Point-cloud publication is controlled by constants near the top of v3.1:

```python
PUBLISH_POINTCLOUD = True
POINTCLOUD_TOPIC = "/depth_anything/point_cloud"
POINTCLOUD_FRAME_ID = "base_link_frd"
POINTCLOUD_STRIDE = 5
POINTCLOUD_MIN_DEPTH = 0.2
POINTCLOUD_MAX_DEPTH = 20.0
```

Before conversion, each 322x322 depth image is filtered by its circle/wing
mask from:

```text
ml_gt_generate/masks/322x322/
```

The converter performs vectorized nearest-depth pooling for each stride block,
uses the color from the winning depth pixel, and publishes packed XYZRGB
points. Publication begins after the node has received depth and rectified
color from all four cameras.

The output follows the same convention as `/stereo/point_cloud`:

```text
frame_id: base_link_frd
+X: forward
+Y: right
+Z: down
```

The current v3.1 cloud intentionally reproduces the stereo node's four ideal
90-degree viewing directions and zero camera origins. It does not yet use the
full measured camera-to-LiDAR transforms in
`calib/2026-09-16/cam2-lidar-calib/camera_raw_rect_lidar_transforms.yaml`.

## Run v3.1

The checked-in `ros_stereo.launch.py` currently references the old
`depth_anything_v2_trt_node-v3.py` name. Run v3.1 directly until that launch
file is intentionally switched:

```bash
source /opt/ros/humble/setup.bash
source ~/ego_ws_ros2/install/setup.bash

~/stereo_venv/bin/python3 \
  Depth-Anything-V2/da_v2_ros2_node_fisheye/depth_anything_v2_trt_node-v3.1.py \
  --engine /path/to/batch4_322x322_metric_depth.engine
```

Check publication:

```bash
ros2 topic hz /camera_2/image_rect
ros2 topic hz /camera_2/relative_depth
ros2 topic hz /depth_anything/point_cloud

ros2 topic echo /depth_anything/point_cloud --field header --once
```

In RViz, add a `PointCloud2` display, select
`/depth_anything/point_cloud`, and use `base_link_frd` as the fixed frame (or
provide a TF from another fixed frame to `base_link_frd`). Select `RGB8` as the
color transformer for camera color.

## Snapshot

Save one image or depth topic as raw NPY plus JPG visualization:

```bash
python3 Depth-Anything-V2/tool_general/img_topic_snapshot.py \
  --img-topic /camera_2/relative_depth
```

The snapshot directory defaults to the directory containing the script.
