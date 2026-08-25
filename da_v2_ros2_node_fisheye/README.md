# Depth Anything V2 Fisheye ROS 2 Node

This standalone ROS 2 node runs one shared Depth Anything V2 TensorRT engine
for four fisheye cameras. It does not rectify images or run stereo matching.

## Topics

| Direction | Topic | Type | Encoding | Size |
|---|---|---|---|---|
| Input | `/camera_0/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | any |
| Input | `/camera_1/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | any |
| Input | `/camera_2/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | any |
| Input | `/camera_3/image_raw` | `sensor_msgs/msg/Image` | common 8-bit encodings | any |
| Output | `/camera_0/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 504x280 |
| Output | `/camera_1/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 504x280 |
| Output | `/camera_2/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 504x280 |
| Output | `/camera_3/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` | 504x280 |

Inputs already 504x280 are used directly. All other input sizes are resized to
504x280 before RGB conversion, ImageNet normalization, and NCHW conversion.
Both dimensions are divisible by the VITS patch size 14, so no padding or crop
is performed.

Each camera has one latest-frame slot. A single GPU worker processes pending
frames in round-robin order, preventing stale queues and camera starvation.
The four cameras are not synchronized.

Relative depth is not metric distance. Nearer objects generally have larger
values, while farther objects generally have smaller values.

## Run

The launch file uses `~/stereo_venv/bin/python3` and the existing dynamic
TensorRT engine by default:

```bash
ros2 launch Depth-Anything-V2/da_v2_ros2_node_fisheye/ros_stereo.launch.py
```

Override either path when needed:

```bash
ros2 launch Depth-Anything-V2/da_v2_ros2_node_fisheye/ros_stereo.launch.py \
  python_executable:=/path/to/python3 \
  engine:=/path/to/depth_anything_v2.engine
```

The runtime log reports one camera inference every two seconds:

```text
camera_0 depth: resized=yes, min=..., max=..., mean=..., nonzero=... |
prep=... ms, infer=... ms, pub=... ms, total=... ms
```

## Snapshot

Save one input or relative-depth topic as raw NPY plus fixed-scale and
normalized JPG files:

```bash
python3 Depth-Anything-V2/da_v2_ros2_node_fisheye/img_topic_snapshot.py \
  --img-topic /camera_0/relative_depth
```

The snapshot directory defaults to the directory containing the script.
