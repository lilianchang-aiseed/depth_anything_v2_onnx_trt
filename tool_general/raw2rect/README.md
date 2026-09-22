# raw2rect

Offline ROS 2 bag conversion from AR0234 Double-Sphere raw images to a 322x322
circular-FOV rectification.
No `ros2 bag play` or running ROS graph is required.

```bash
cd ~/tools_rectify_stereo/Depth-Anything-V2
source ~/.bashrc
conda activate aiseed1

python3 tool_general/raw2rect/raw2rect_bag.py \
  --input "$COMMON_SHARE/bags/nx-2.0/0917/flight_data_2026_09_17-15_31_25/flight_data_2026_09_17-15_31_25_0.mcap" \
  --out-dir /tmp/flight_data_0917_rectified
```

The output bag keeps all original topics. By default, an existing configured
`/camera_<n>/image_rect` topic is copied unchanged and only missing rect topics
are generated. To regenerate and replace all configured rect topics, add:

```bash
--replace-old-rect
```

Newly calibrated rectified images use the raw image's bag timestamp and
`header`.

The default 90-degree full FOV is a circular viewing cone. Pixels outside the
cone, including the square corners, are black.

The output metadata uses rosbag format version 8 for ROS 2 Humble
compatibility.

For a short structural smoke test, add `--max-bag-messages 1000`. The output
directory must not already exist; the tool never overwrites a bag.
