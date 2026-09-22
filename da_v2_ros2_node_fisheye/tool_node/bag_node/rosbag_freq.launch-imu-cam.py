# remember !! set stereo topics to record

import os
import sys
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


# Original non-camera/LiDAR topics recorded directly.
# No topic_tools throttle nodes are created.
OTHER_TOPICS_TO_RECORD = [

    # cube / imu
    '/diagnostics',
    # '/mavros/global_position/compass_hdg',
    # '/mavros/global_position/global',
    # '/mavros/global_position/raw/fix',
    # '/mavros/global_position/raw/gps_vel',
    # '/mavros/local_position/accel',
    # '/mavros/local_position/odom',
    # '/mavros/local_position/pose',
    # '/mavros/local_position/velocity_body',
    # '/mavros/local_position/velocity_local',
    '/mavros/imu/data',
    '/mavros/imu/data_raw',
    '/mavros/imu/mag',
    '/mavros/time_reference',
    '/tf',
    '/tf_static',

    # rectified image
    # '/stereo_0_3/left/image_rect',
    # '/stereo_0_3/right/image_rect',
    # '/stereo_1_0/left/image_rect',
    # '/stereo_1_0/right/image_rect',
    # '/stereo_2_1/left/image_rect',
    # '/stereo_2_1/right/image_rect',
    # '/stereo_3_2/left/image_rect',
    # '/stereo_3_2/right/image_rect',

    '/camera_0/image_rect',
    '/camera_1/image_rect',
    '/camera_2/image_rect',
    '/camera_3/image_rect',

    '/camera_0/relative_depth',
    '/camera_1/relative_depth',
    '/camera_2/relative_depth',
    '/camera_3/relative_depth',

    '/depth_anything/point_cloud',

    # raw image 
    '/camera_0/image_raw', 
    '/camera_1/image_raw', 
    '/camera_2/image_raw', 
    '/camera_3/image_raw', 

    '/camera_0/image_info', 
    '/camera_1/image_info', 
    '/camera_2/image_info', 
    '/camera_3/image_info', 
    # '/camera/camera/color/image_raw',

    # realsense
    # '/d435/d435_node/infra1/image_rect_raw',
    '/d455/d455_node/depth/image_rect_raw',

]

NUM_CAMERAS = 4

LIDAR_TOPICS = {
    'hesai': {
        'points': '/lidar_points',
        'imu': '/lidar_imu',
        'loss': '/lidar_packets_loss',
    },
    'livox': {
        'points': '/livox/lidar',
        'imu': '/livox/imu',
        'loss': '',
    },
}


def _as_bool(value):
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    snapshot_freq = float(
        LaunchConfiguration('freq').perform(context)
    )
    lidar_vendor = LaunchConfiguration('lidar_vendor').perform(context).lower()
    lidar_sync = _as_bool(LaunchConfiguration('lidar_sync').perform(context))

    if lidar_vendor not in LIDAR_TOPICS:
        choices = ', '.join(sorted(LIDAR_TOPICS))
        raise ValueError(
            f'Unsupported lidar_vendor={lidar_vendor!r}; choose one of: {choices}'
        )

    lidar_topics = LIDAR_TOPICS[lidar_vendor]

    home_path = os.path.expanduser('~')
    workspace_root = os.path.join(home_path, 'ego_ws_ros2')
    bags_path = os.path.join(workspace_root, 'bags')

    os.makedirs(bags_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    bag_directory = os.path.join(
        bags_path,
        f'flight_data_{timestamp}',
    )

    # Record the original camera topics directly.
    # camera_topics = [
    #     topic
    #     for camera_index in range(NUM_CAMERAS)
    #     for topic in (
    #         # f'/camera_{camera_index}/image_raw/compressed',
    #         # f'/camera_{camera_index}/image_raw',
    #         # f'/camera_{camera_index}/camera_info',
            
    #     )
    # ]

    # LiDAR-driven synchronization node.
    snapshot_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'tool_general', 'lidar_snapshot_node.py',
    )
    snapshot_script = os.path.abspath(snapshot_script)

    snapshot_action = ExecuteProcess(
        cmd=[
            sys.executable,
            snapshot_script,
            '--ros-args',
            '-p',
            f'freq:={snapshot_freq}',
            '-p',
            f'lidar_points_topic:={lidar_topics["points"]}',
            '-p',
            f'lidar_imu_topic:={lidar_topics["imu"]}',
            '-p',
            f'lidar_loss_topic:={lidar_topics["loss"]}',
            '-p',
            f'enable_loss_packet:={str(bool(lidar_topics["loss"])).lower()}',
        ],
        output='screen',
        shell=False,
    )

    # Record original topics directly.
    # No throttle node and no /throttled namespace are used.
    topics_to_record = [
        *OTHER_TOPICS_TO_RECORD,
        lidar_topics['points'],
        lidar_topics['imu'],
    ]

    if lidar_topics['loss']:
        topics_to_record.append(lidar_topics['loss'])

    if lidar_sync:
        topics_to_record.extend([
            '/synced/lidar_points',
            '/synced/lidar_imu',
            *[
                topic
                for camera_index in range(NUM_CAMERAS)
                for topic in (
                    f'/synced/camera_{camera_index}/image_raw',
                    f'/synced/camera_{camera_index}/camera_info',
                )
            ],
        ])
        if lidar_topics['loss']:
            topics_to_record.append('/synced/lidar_packets_loss')

    # Preserve order while removing accidental duplicate topic names.
    topics_to_record = list(dict.fromkeys(topics_to_record))

    record_action = ExecuteProcess(
        cmd=[
            'ros2',
            'bag',
            'record',
            '--storage',
            'mcap',
            # *camera_topics,
            *topics_to_record,
            '-o',
            bag_directory,
        ],
        output='screen',
        shell=False,
    )

    actions = [record_action]
    if lidar_sync:
        actions.insert(0, snapshot_action)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'freq',
            default_value='0.5',
            description=(
                'Frequency parameter passed to lidar_snapshot_node. '
                'MAVROS, TF, and camera topics are recorded at their '
                'original frequencies.'
            ),
        ),
        DeclareLaunchArgument(
            'lidar_vendor',
            default_value='hesai',
            description=(
                'LiDAR driver topic layout: hesai uses /lidar_points and '
                '/lidar_imu; livox uses /livox/lidar and /livox/imu.'
            ),
        ),
        DeclareLaunchArgument(
            'lidar_sync',
            default_value='false',
            description=(
                'When true, also run lidar_snapshot_node and record /synced/* '
                'topics. Raw LiDAR topics are always recorded.'
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
