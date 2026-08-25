"""Launch four-camera fisheye Depth Anything V2 TensorRT inference.

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
DEFAULT_PYTHON = Path.home() / "stereo_venv" / "bin" / "python3"
DEFAULT_ENGINE = (
    HERE.parent / "checkpoints" / "depth_anything_v2_vits_dynamic.engine"
)


def generate_launch_description() -> LaunchDescription:
    python_executable = LaunchConfiguration("python_executable")
    engine = LaunchConfiguration("engine")

    depth_anything = ExecuteProcess(
        cmd=[
            python_executable, 
            str(NODE_SCRIPT), 
            "--engine", 
            engine,
            "--no-refine"],
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
            depth_anything,
        ]
    )
