#!/usr/bin/env python3
import cv2
import time
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray, Float32MultiArray
from .rknnpool import rknnPoolExecutor
from .func import myFunc, CLASSES

class CrossroadDetectorNode(Node):
    def __init__(self):
        super().__init__('crossroad_detector_node')
        # 常量
        self.MODEL_PATH = '/home/orangepi/car_ros2/src/rknn_yolo/rknn_yolo/rknnModel/num.rknn'
        self.THREADS = 3
        self.SHOW_GUI = True
        self.MIN_CONTOUR_AREA = 10
        self.IMAGE_WIDTH = 640

        # 状态变量
        self.target_wards = []  # 目标病房列表
        self.mode = 1
        self.turn_direction = 0

        # ROS2 发布与订阅
        self.turn_direction_pub = self.create_publisher(Int8MultiArray, '/direction', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.ward_sub = self.create_subscription(Int8MultiArray, '/target_wards', self.update_target_wards, 10)

        # 初始化
        self.bridge = CvBridge()
        self.pool = rknnPoolExecutor(rknnModel=self.MODEL_PATH, TPEs=self.THREADS, func=myFunc)

        self.get_logger().info('转向检测节点已启动')

    def update_mode(self, msg):
        self.mode = int(msg.data[2])

    def update_target_wards(self, msg):
        self.target_wards = list(msg.data)
        #self.get_logger().info(f'更新目标病房: {self.target_wards}')

    def process_red_line(self, image):
        h, w = image.shape[:2]
        roi = image[h//2:h, :, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([155, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > self.MIN_CONTOUR_AREA:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(max_contour)
                    return center_x, contours, mask, (x, y, w, h)
        return None, None, mask, None

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {str(e)}')
            return

        red_line_center_x, contours, binary, red_line_box = self.process_red_line(frame)
        self.pool.put(frame)
        yolo_result, flag = self.pool.get()
        if not flag:
            self.get_logger().warn('YOLO 处理失败')
            yolo_result = (frame, None, None, None, None, None)
        processed_frame, boxes, detected_classes, scores, class_to_center_x = yolo_result
        detected_number = None
        self.turn_direction = 0

        # 检查 detected_classes 是否包含目标病房
        if detected_classes and len(detected_classes) > 0 and detected_classes != ['finall']:
            #self.get_logger().info(f'检测到的类别: {detected_classes}')
            # 将 detected_classes 转换为数字形式（CLASSES 已经是数字字符串格式，如 '3'）
            detected_numbers = []
            for cls in detected_classes:
                try:
                    number = int(cls)  # 直接转换为整数，因为 CLASSES 是 '1', '2', ..., '8'
                    detected_numbers.append(number)
                except (ValueError, IndexError):
                    #self.get_logger().warn(f'无法解析类别 {cls} 为病房数字')
                    continue

            # 检查是否有目标病房
            matched_wards = [num for num in detected_numbers if num in self.target_wards]
            if matched_wards:
                detected_number = matched_wards[0]  # 选择第一个匹配的病房
                #self.get_logger().info(f'匹配到目标病房: {detected_number}')
                # 获取对应数字的 number_center_x
                number_center_x_for_detected = class_to_center_x.get(str(detected_number))
                if red_line_center_x is not None and number_center_x_for_detected is not None:
                    if number_center_x_for_detected < red_line_center_x:
                        self.turn_direction = -1
                        self.get_logger().info(f'检测到病房 {detected_number}，数字在红线左边，决定左转')
                    else:
                        self.turn_direction = 1
                        self.get_logger().info(f'检测到病房 {detected_number}，数字在红线右边，决定右转')
                    turn_msg = Int8MultiArray()
                    turn_msg.data = [self.turn_direction]
                    self.turn_direction_pub.publish(turn_msg)

        if red_line_box:
            x, y, w, h = red_line_box
            y += frame.shape[0] // 2
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if red_line_center_x is not None:
                cv2.circle(processed_frame, (red_line_center_x, y + h // 2), 5, (0, 0, 255), -1)

        cv2.putText(processed_frame, f'Turn: {"Left" if self.turn_direction == -1 else "Right" if self.turn_direction == 1 else "None"}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'Number: {detected_number if detected_number else "None"}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, f'Target Wards: {self.target_wards}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f'Detected Classes: {detected_classes}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # if self.SHOW_GUI:
        #     cv2.imshow('Crossroad Detection', processed_frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         self.destroy_node()

    def destroy_node(self):
        self.get_logger().info('关闭节点')
        self.pool.release()
        if self.SHOW_GUI:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CrossroadDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()