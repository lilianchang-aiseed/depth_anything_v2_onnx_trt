"""Launch multi-pair stereo rectification and Depth Anything V2 TensorRT.

This one launch file starts both local scripts:
  1. multi_rectify_node.py publishes rectified stereo images.
  2. depth_anything_v2_trt_node.py consumes them and publishes relative depth.

Usage:
    ros2 launch ./ros_stereo.launch.py

By default the node uses the same Python environment as the original stereo
program:
    ~/stereo_venv/bin/python3

Use a different TensorRT engine:
    ros2 launch ./ros_stereo.launch.py engine:=/path/to/model.engine
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


HERE = Path(__file__).resolve().parent
NODE_SCRIPT = HERE / "depth_anything_v2_trt_node.py"
STEREO_NODE_SCRIPT = HERE / "multi_rectify_node.py"
DEFAULT_PYTHON = Path.home() / "stereo_venv" / "bin" / "python3"
DEFAULT_ENGINE = (
    HERE.parent / "checkpoints" / "depth_anything_v2_vits_dynamic.engine"
)


def generate_launch_description() -> LaunchDescription:
    python_executable = LaunchConfiguration("python_executable")
    engine = LaunchConfiguration("engine")

    stereo = ExecuteProcess(
        cmd=[python_executable, str(STEREO_NODE_SCRIPT)],
        output="screen",
        shell=False,
    )

    depth_anything = ExecuteProcess(
        cmd=[python_executable, str(NODE_SCRIPT), "--engine", engine],
        output="screen",
        shell=False,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "python_executable",
                default_value=str(DEFAULT_PYTHON),
                description=(
                    "Python executable with ROS 2, CUDA PyTorch, and TensorRT"
                ),
            ),
            DeclareLaunchArgument(
                "engine",
                default_value=str(DEFAULT_ENGINE),
                description="Depth Anything V2 TensorRT engine path",
            ),
            stereo,
            depth_anything,
        ]
    )
