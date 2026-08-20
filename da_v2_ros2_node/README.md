# Depth Anything V2 ROS 2 Node

This standalone ROS 2 Python node subscribes to the left and right rectified
images for four fixed stereo pairs and runs the local dynamic Depth Anything
V2 VITS TensorRT engine. It is not packaged as an `ament_python` ROS 2 package
and does not use ONNX Runtime.

## Topics

The fixed pairs are `0_3`, `1_0`, `2_1`, and `3_2`. For every pair, the node
uses these topics:

| Direction | Topic pattern | Type | Encoding |
|---|---|---|---|
| Input | `/stereo_<l>_<r>/left/image_rect` | `sensor_msgs/msg/Image` | `bgr8` |
| Input | `/stereo_<l>_<r>/right/image_rect` | `sensor_msgs/msg/Image` | `bgr8` |
| Output | `/stereo_<l>_<r>/left/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` |
| Output | `/stereo_<l>_<r>/right/relative_depth` | `sensor_msgs/msg/Image` | `32FC1` |

Each output preserves its input timestamp, frame ID, height, and width. The
`32FC1` output is relative depth, not metric depth in metres. Rectified images
remain published by the stereo node; this node does not republish them.

Each of the eight inputs has one latest-frame slot. A single GPU worker visits
pending slots in round-robin order, so old frames do not accumulate and one
high-rate input cannot starve the others.

The stereo node currently publishes 320x320 images. Because the VITS patch size
is 14, this node pads the right and bottom edges to 322x322 using replicated
pixels, runs inference, and crops the prediction back to 320x320. It does not
resize or stretch the rectified input geometry.

## Run

```bash
source /opt/ros/<distro>/setup.bash
cd /home/r12543040/tools_rectify_stereo

python3 Depth-Anything-V2/da_v2_ros2_node/depth_anything_v2_trt_node.py
```

Or launch it through ROS 2:

```bash
ros2 launch Depth-Anything-V2/da_v2_ros2_node/ros_stereo.launch.py
```

The launch file uses `~/stereo_venv/bin/python3` by default, matching the
original stereo program. Override `python_executable` only when needed.

This launch file starts both `multi_rectify_node.py` and
`depth_anything_v2_trt_node.py`. One launch therefore publishes the rectified
stereo images and their Depth Anything relative-depth outputs.

The rectification node does not load OpenStereo or LightStereo and does not
publish stereo disparity or point clouds.

The default model is:

```text
Depth-Anything-V2/checkpoints/depth_anything_v2_vits_dynamic.engine
```

Use another TensorRT engine with:

```bash
python3 Depth-Anything-V2/da_v2_ros2_node/depth_anything_v2_trt_node.py \
  --engine /path/to/depth_anything_v2.engine
```

Test the engine without ROS using the bundled demo image:

```bash
python3 Depth-Anything-V2/da_v2_ros2_node/depth_anything_v2_trt_node.py \
  --test-image Depth-Anything-V2/assets/examples/demo01.jpg \
  --test-output /tmp/depth_trt_test.png
```

The test prints output min/max/mean/nonzero statistics and saves both a color
visualization (`.png`) and the raw `float32` relative depth (`.npy`).

ROS remapping arguments are preserved:

```bash
python3 Depth-Anything-V2/da_v2_ros2_node/depth_anything_v2_trt_node.py \
  --ros-args \
  -r /stereo_1_0/left/image_rect:=/another/image_rect
```

## Verify

```bash
ros2 topic info /stereo_1_0/left/relative_depth
ros2 topic echo /stereo_1_0/left/relative_depth --once | \
  grep -E 'height|width|encoding'
```

Expected for the current stereo input:

```text
height: 320
width: 320
encoding: 32FC1
```

## Save One ROS Image

Automatically decode a common color, grayscale, integer-depth, or float image
topic and save both raw NPY data and a displayable JPG:

```bash
python3 Depth-Anything-V2/da_v2_ros2_node/save_ros_image.py \
  --img-topic /stereo_0_3/left/relative_depth
```

The output directory defaults to the directory containing the script. Select
another directory with `--out-dir /tmp`.
