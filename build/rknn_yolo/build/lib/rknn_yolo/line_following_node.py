#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray, Int8MultiArray, Float32, Int8
import time
from rknn_yolo.lib.pid_controller import PIDController
from rknn_yolo.lib.image_processor import process_image
from rknn_yolo.lib.motion_controller import MotionController
from rknn_yolo.lib.handle_logic import handle_logic

class LineFollowingNode(Node):
    def __init__(self):
        super().__init__('line_following_node')
        self.bridge = CvBridge()
        self.pid = PIDController(kp=0.50, ki=0.0, kd=0.03)
        self.motion = MotionController(self)
        self.handler = handle_logic(self)

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.crossroad_sub = self.create_subscription(Int8, '/crossroad_id', self.motion.crossroad_callback, 10)
        self.direction_sub = self.create_subscription(Int8MultiArray, '/ward_directions', self.motion.direction_callback, 10)
        self.mode_sub = self.create_subscription(Float32, '/serial_mode', self.motion.update_mode, 10)
        self.ward_sub = self.create_subscription(Int8MultiArray, '/target_wards', self.motion.update_target_wards, 10)
        self.cmd_vel_pub = self.create_publisher(Float32MultiArray, '/cmd_vel', 10)
        self.reset = self.create_publisher(Int8, '/reset', 10)

        self.get_logger().info('线跟踪节点已启动')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {str(e)}')
            return

        line_center, center_y, binary, bounding_box = process_image(frame, self.motion.turning, self.motion.turn_direction)

        self.get_logger().info(f'当前状态: {self.motion.state}, 路口ID: {self.motion.current_crossroad_id}, 目标病房: {self.motion.target_wards}, 当前病房索引: {self.motion.current_ward_index}')

        if self.motion.waiting_for_direction:
            elapsed_time = time.time() - self.motion.direction_check_start_time
            if elapsed_time < self.motion.DIRECTION_CHECK_DELAY:
                self.get_logger().info(f'等待方向信息更新，剩余时间: {self.motion.DIRECTION_CHECK_DELAY - elapsed_time:.2f}秒')
                left_speed, right_speed = self.motion.BASE_SPEED, self.motion.BASE_SPEED
                self.motion.publish_cmd_vel(left_speed, right_speed)
                return
            else:
                self.handler.handle_crossroad_logic()

        left_speed, right_speed = self.handler.handle_state_logic(line_center)

        binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(binary_display, (x, y), (x + w, y + h), (0, 0, 255), 2)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()