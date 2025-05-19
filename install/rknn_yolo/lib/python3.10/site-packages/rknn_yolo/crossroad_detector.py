#!/usr/bin/env python3
import cv2
import time
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray, Int8
from .rknnpool import rknnPoolExecutor
from .func import myFunc

class CrossroadDetectorNode(Node):
    def __init__(self):
        super().__init__('crossroad_detector')
        # 常量
        self.MODEL_PATH = '/home/orangepi/car_ros2/src/rknn_yolo/rknn_yolo/rknnModel/num.rknn'
        self.THREADS = 3
        self.SHOW_GUI = True
        self.MIN_CONTOUR_AREA = 10
        self.IMAGE_WIDTH = 640
        self.CROSSROAD_COOLDOWN = 2.5  # 十字路口检测冷却时间（秒）

        # 状态变量
        self.target_wards = []
        self.crossroad_id = 0
        self.last_crossroad_id = 0
        self.last_crossroad_time = 0
        self.detected_wards = {}
        self.pending_wards = {}  # 临时存储检测到的病房信息，等待十字路口确认

        # ROS2 发布与订阅
        self.crossroad_pub = self.create_publisher(Int8, '/crossroad_id', 10)
        self.direction_pub = self.create_publisher(Int8MultiArray, '/ward_directions', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.ward_sub = self.create_subscription(Int8MultiArray, '/target_wards', self.update_target_wards, 10)
        self.reset = self.create_subscription(Int8, '/reset', self.update_reset, 10)

        # 初始化
        self.bridge = CvBridge()
        self.pool = rknnPoolExecutor(rknnModel=self.MODEL_PATH, TPEs=self.THREADS, func=myFunc)

        self.get_logger().info('十字路口检测节点已启动')

    def update_target_wards(self, msg):
        self.target_wards = list(msg.data)
        #self.get_logger().info(f'更新目标病房: {self.target_wards}')
    def update_reset(self, msg):
        self.reset = msg.data

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

    def detect_crossroad(self, frame):
        h, w = frame.shape[:2]
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([155, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        # 上部区域检测
        roi_upper = frame[0:int(h*0.2), int(w * 0.2):int(w * 0.8), :]
        hsv_upper = cv2.cvtColor(roi_upper, cv2.COLOR_BGR2HSV)
        mask1_upper = cv2.inRange(hsv_upper, lower_red1, upper_red1)
        mask2_upper = cv2.inRange(hsv_upper, lower_red2, upper_red2)
        mask_upper = mask1_upper | mask2_upper
        kernel = np.ones((5, 5), np.uint8)
        mask_upper = cv2.morphologyEx(mask_upper, cv2.MORPH_OPEN, kernel, iterations=2)
        contours_upper, _ = cv2.findContours(mask_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 左侧区域检测
        start_col = int(h * 0.2)
        end_col = int(h * 0.6)
        roi_left = frame[start_col:end_col, 0:int(w*0.2), :]
        hsv_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2HSV)
        mask1_left = cv2.inRange(hsv_left, lower_red1, upper_red1)
        mask2_left = cv2.inRange(hsv_left, lower_red2, upper_red2)
        mask_left = mask1_left | mask2_left
        mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_OPEN, kernel, iterations=2)
        contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 右侧区域检测
        roi_right = frame[start_col:end_col, int(w*0.8):w, :]
        hsv_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2HSV)
        mask1_right = cv2.inRange(hsv_right, lower_red1, upper_red1)
        mask2_right = cv2.inRange(hsv_right, lower_red2, upper_red2)
        mask_right = mask1_right | mask2_right
        mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_OPEN, kernel, iterations=2)
        contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def has_significant_contour(contours, min_area=10, min_width_ratio=0.05):
            if not contours:
                return False
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                if area > min_area and w_contour > w * min_width_ratio:
                    return True
            return False

        upper_detected = has_significant_contour(contours_upper)
        left_detected = has_significant_contour(contours_left)
        right_detected = has_significant_contour(contours_right)

        is_crossroad = (
            (upper_detected and right_detected) or
            (upper_detected and left_detected) or
            (left_detected and right_detected)
        )

        return is_crossroad, left_detected, right_detected

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {str(e)}')
            return

        # 检测十字路口
        current_time = time.time()
        is_crossroad, left_detected, right_detected = self.detect_crossroad(frame)
        if self.reset == 0:
            self.crossroad_id = 0
            self.last_crossroad_id = 0
            self.last_crossroad_time = 0
            self.detected_wards.clear()
            self.pending_wards.clear()
            self.get_logger().info('检测到重置信号，清空十字路口信息')
            self.reset = 1
        if is_crossroad and (current_time - self.last_crossroad_time) > self.CROSSROAD_COOLDOWN:
            self.crossroad_id += 1
            self.last_crossroad_id = self.crossroad_id
            self.last_crossroad_time = current_time
            self.get_logger().info(f'检测到十字路口 {self.crossroad_id}')
            crossroad_msg = Int8()
            crossroad_msg.data = self.crossroad_id
            self.crossroad_pub.publish(crossroad_msg)

            # 如果有待发布的病房信息，立即发布
            if self.pending_wards:
                ward_directions = self.pending_wards
                self.detected_wards[self.crossroad_id] = ward_directions
                direction_msg = Int8MultiArray()
                num_wards = len(ward_directions)
                data = [self.crossroad_id, num_wards]
                for ward, direction in ward_directions.items():
                    data.extend([ward, direction])
                direction_msg.data = data
                self.direction_pub.publish(direction_msg)
                self.get_logger().info(f'发布转向信息: {direction_msg.data}')
                self.pending_wards = {}  # 清空待发布信息

        # 处理红线
        red_line_center_x, contours, binary, red_line_box = self.process_red_line(frame)
        self.pool.put(frame)
        yolo_result, flag = self.pool.get()
        if not flag:
            self.get_logger().warn('YOLO 处理失败')
            yolo_result = (frame, None, None, None, None, None)
        processed_frame, boxes, detected_classes, scores, class_to_center_x = yolo_result
        detected_number = None
        turn_direction = 0

        # 检查 detected_classes 是否包含目标病房
        ward_directions = {}
        if detected_classes and len(detected_classes) > 0 and detected_classes != ['finall']:
            detected_numbers = []
            for cls in detected_classes:
                try:
                    number = int(cls)
                    detected_numbers.append(number)
                except (ValueError, IndexError):
                    self.get_logger().warn(f'无法解析类别 {cls} 为病房数字')
                    continue

            matched_wards = [num for num in detected_numbers if num in self.target_wards]
            if matched_wards:
                for ward in matched_wards:
                    number_center_x = class_to_center_x.get(str(ward))
                    if red_line_center_x is not None and number_center_x is not None:
                        if number_center_x < red_line_center_x:
                            turn_direction = -1
                            self.get_logger().info(f'优先选择左侧病房 {ward}，转向: -1')
                        else:
                            turn_direction = 1
                            self.get_logger().info(f'优先选择右侧病房 {ward}，转向: 1')
                        ward_directions[ward] = turn_direction

        # 仅在检测到十字路口后发布方向信息，否则暂存
        if ward_directions:
            if self.crossroad_id == self.last_crossroad_id:
                self.detected_wards[self.crossroad_id] = ward_directions
                direction_msg = Int8MultiArray()
                num_wards = len(ward_directions)
                data = [self.crossroad_id, num_wards]
                for ward, direction in ward_directions.items():
                    data.extend([ward, direction])
                direction_msg.data = data
                self.direction_pub.publish(direction_msg)
                self.get_logger().info(f'发布转向信息: {direction_msg.data}')
            else:
                self.pending_wards.update(ward_directions)
                self.get_logger().info(f'暂存病房方向信息: {ward_directions}，等待十字路口 {self.crossroad_id + 1}')

        # 可视化
        if red_line_box:
            x, y, w, h = red_line_box
            y += frame.shape[0] // 2
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if red_line_center_x is not None:
                cv2.circle(processed_frame, (red_line_center_x, y + h // 2), 5, (0, 0, 255), -1)

        cv2.putText(processed_frame, f'Turn: {"Left" if turn_direction == -1 else "Right" if turn_direction == 1 else "None"}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'Number: {detected_number if detected_number else "None"}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, f'Target Wards: {self.target_wards}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f'Detected Classes: {detected_classes}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.SHOW_GUI:
            cv2.imshow('Crossroad Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy_node()

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