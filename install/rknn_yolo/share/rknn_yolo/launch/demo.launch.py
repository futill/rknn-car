from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rknn_yolo',
            executable='camre',
            name='camre',
            output='screen'
        ),
        # Node(
        #     package='rknn_yolo',
        #     executable='opencv_pid',
        #     name='opencv_pid',
        #     output='screen'
        # ),
        # Node(
        #     package='rknn_yolo',
        #     executable='rknn_yolo',
        #     name='rknn_yolo',
        #     output='screen'
        # ),
        Node(
            package='rknn_yolo',
            executable='line_following_node',
            name='line_following_node',
            output='screen'
        ),
        Node(
            package='rknn_yolo',
            executable='crossroad_detector',
            name='crossroad_detector',
            output='screen'
        ),
        Node(
            package='rknn_yolo',
            executable='motor',
            name='motor',
            output='screen'
        ),
        Node(
            package='rknn_yolo',
            executable='targer',
            name='targer',
            output='screen'
        )
    ])