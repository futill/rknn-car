#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray, Int8MultiArray, Float32, Int8

class PIDController:
    def __init__(self, kp=0.26, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def process_image(image, is_turning=False, turn_direction=0):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([155, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    line_center = None
    center_y = None
    bounding_box = None
    current_mask = None

    roi_line = image[int(h*0.75):h, :, :]
    h_roi_line, w_roi_line = roi_line.shape[:2]
    hsv_line = cv2.cvtColor(roi_line, cv2.COLOR_BGR2HSV)
    mask1_line = cv2.inRange(hsv_line, lower_red1, upper_red1)
    mask2_line = cv2.inRange(hsv_line, lower_red2, upper_red2)
    mask_line = mask1_line | mask2_line
    mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, kernel)
    contours_line, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not is_turning:
        current_mask = mask_line
        if contours_line:
            max_contour = max(contours_line, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 100:
                fixed_y_roi = h_roi_line // 2
                center_y = int(h * 0.875)
                horizontal_line = mask_line[fixed_y_roi, :]
                white_pixels = np.where(horizontal_line > 0)[0]
                if len(white_pixels) > 0:
                    line_center = int(np.mean(white_pixels))
                    x, y, w, h = cv2.boundingRect(max_contour)
                    if w > 20:
                        y += int(h*0.75)
                        bounding_box = (x, y, w, h)
    else:
        roi_turn = image[0:int(h*0.5), :, :]
        h_roi_turn, w_roi_turn = roi_turn.shape[:2]
        hsv_turn = cv2.cvtColor(roi_turn, cv2.COLOR_BGR2HSV)
        mask1_turn = cv2.inRange(hsv_turn, lower_red1, upper_red1)
        mask2_turn = cv2.inRange(hsv_turn, lower_red2, upper_red2)
        mask_turn = mask1_turn | mask2_turn
        mask_turn = cv2.morphologyEx(mask_turn, cv2.MORPH_OPEN, kernel)
        contours_turn, _ = cv2.findContours(mask_turn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_mask = mask_turn

        if contours_turn:
            max_contour = max(contours_turn, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 100:
                fixed_y_roi = h_roi_turn // 2
                center_y = int(h * 0.25)
                if turn_direction == -1:
                    horizontal_line = mask_turn[fixed_y_roi, 0:w_roi_turn//2]
                    white_pixels = np.where(horizontal_line > 0)[0]
                elif turn_direction == 1:
                    horizontal_line = mask_turn[fixed_y_roi, w_roi_turn//2:w_roi_turn]
                    white_pixels = np.where(horizontal_line > 0)[0] + w_roi_turn//2
                else:
                    horizontal_line = mask_turn[fixed_y_roi, :]
                    white_pixels = np.where(horizontal_line > 0)[0]

                if len(white_pixels) > 0:
                    line_center = int(np.mean(white_pixels))
                    x, y, w, h = cv2.boundingRect(max_contour)
                    bounding_box = (x, y, w, h)

    return line_center, center_y, current_mask, bounding_box

class LineFollowingNode(Node):
    def __init__(self):
        super().__init__('line_following_node')
        self.bridge = CvBridge()

        self.IMAGE_WIDTH = 640
        self.BASE_SPEED = 90.0
        self.MAX_SPEED = 120.0
        self.KP = 0.26
        self.KI = 0.0
        self.KD = 0.1
        self.MAX_LINE_LOST = 2
        self.UNLOAD_TIME = 1.0
        self.TURN_TIME = 0.85
        self.TURN_AROUND_TIME = 1
        self.CENTER_THRESHOLD = 50
        self.WAIT_AT_CROSSROAD_TIME = 0.0
        self.STRAIGHT_WAIT_TIME = 0.5
        self.DIRECTION_CHECK_DELAY = 0.17

        self.STATES = {
            'WAITING': 1,
            'TO_WARD': 2,
            'UNLOADING': 3,
            'TURNING_AROUND': 4,
            'TO_SECOND_WARD': 5,
            'RETURNING': 6
        }
        self.state = self.STATES['WAITING']
        self.mode = 1
        self.target_wards = []
        self.crossroad_history = []
        self.current_crossroad_id = 0
        self.current_ward_index = 0
        self.turn_history = []
        self.return_turn_index = 0
        self.last_crossroad_id = 0
        self.start_time = None
        self.unload_start_time = None
        self.turn_around_start_time = None
        self.enter_crossroad_time = None
        self.straight_start_time = None
        self.direction_check_start_time = None
        self.turning = False
        self.forcing_straight = False
        self.waiting_for_direction = False
        self.waiting_at_crossroad = False
        self.is_straight_waiting = False
        self.entered_crossroad = False
        self.line_left_center = False
        self.line_lost_count = 0
        self.turn_direction = 0
        self.image_center = self.IMAGE_WIDTH // 2
        self.pid = PIDController(self.KP, self.KI, self.KD)
        self.ward_directions = {}
        self.crossroad_4_passed = False
        self.wards_same_side = False  # 新增标志：两个病房是否在路口 3 同一侧

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.crossroad_sub = self.create_subscription(Int8, '/crossroad_id', self.crossroad_callback, 10)
        self.direction_sub = self.create_subscription(Int8MultiArray, '/ward_directions', self.direction_callback, 10)
        self.mode_sub = self.create_subscription(Float32, '/serial_mode', self.update_mode, 10)
        self.ward_sub = self.create_subscription(Int8MultiArray, '/target_wards', self.update_target_wards, 10)
        self.cmd_vel_pub = self.create_publisher(Float32MultiArray, '/cmd_vel', 10)

        self.get_logger().info('线跟踪节点已启动')

    def crossroad_callback(self, msg):
        if self.turning or self.waiting_at_crossroad:
            self.get_logger().info(f'当前处于转弯或等待路口中心状态，忽略十字路口ID更新: {msg.data}')
            return

        previous_crossroad_id = self.current_crossroad_id
        self.current_crossroad_id = msg.data
        if self.current_crossroad_id != self.last_crossroad_id:
            self.last_crossroad_id = self.current_crossroad_id
            self.waiting_for_direction = True
            self.direction_check_start_time = time.time()
            self.get_logger().info(f'收到十字路口ID: {self.current_crossroad_id}（前一个ID: {previous_crossroad_id}）')

    def direction_callback(self, msg):
        crossroad_id = msg.data[0]
        num_wards = msg.data[1]
        ward_directions = {}
        for i in range(num_wards):
            ward = msg.data[2 + i*2]
            direction = msg.data[3 + i*2]
            ward_directions[ward] = direction
        self.ward_directions[crossroad_id] = ward_directions
        self.get_logger().info(f'收到病房方向信息: 十字路口 {crossroad_id}, 方向: {ward_directions}')

        # 判断路口 3 的两个病房是否在同一侧
        if crossroad_id == 3 and len(self.target_wards) >= 2 and self.wards_same_side==False:
            ward1, ward2 = self.target_wards[0], self.target_wards[1]
            if ward1 in ward_directions and ward2 in ward_directions:
                self.wards_same_side = (ward_directions[ward1] == ward_directions[ward2])
                self.get_logger().info(f'路口 3 病房方向判断: 病房 {ward1} 方向 {ward_directions[ward1]}, 病房 {ward2} 方向 {ward_directions[ward2]}, 同一侧: {self.wards_same_side}')

    def update_mode(self, msg):
        self.quality = msg.data
        if self.state == self.STATES['WAITING']:
            self.mode = 2 if self.quality >= 200 else 1
            if self.mode == 2:
                if not self.target_wards:
                    self.get_logger().warn('目标病房未设置，等待 /target_wards')
                    self.mode = 1
                    return
                self.state = self.STATES['TO_WARD']
                self.start_time = time.time()
                self.last_crossroad_id = 0
                self.current_ward_index = 0
                self.get_logger().info(f'开始送药，目标病房: {self.target_wards}')
        elif self.state == self.STATES['RETURNING']:
            self.mode = 1
            self.state = self.STATES['WAITING']
            self.last_crossroad_id = 0
            self.current_ward_index = 0
            self.get_logger().info('返回药房完成')

    def update_target_wards(self, msg):
        self.target_wards = list(msg.data)
        self.get_logger().info(f'更新目标病房: {self.target_wards}')
        if self.state == self.STATES['WAITING'] and self.mode == 2:
            self.state = self.STATES['TO_WARD']
            self.start_time = time.time()
            self.last_crossroad_id = 0
            self.current_ward_index = 0
            self.get_logger().info(f'开始送药，目标病房: {self.target_wards}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {str(e)}')
            return

        left_speed = right_speed = 0.0

        if self.state == self.STATES['WAITING']:
            self.get_logger().info('当前状态为 WAITING，停车')
            self.publish_cmd_vel(0.0, 0.0)
            return

        line_center, center_y, binary, bounding_box = process_image(frame, self.turning, self.turn_direction)

        self.get_logger().info(f'当前状态: {self.state}, 路口ID: {self.current_crossroad_id}, 目标病房: {self.target_wards}, 当前病房索引: {self.current_ward_index}')

        if self.waiting_for_direction:
            elapsed_time = time.time() - self.direction_check_start_time
            if elapsed_time < self.DIRECTION_CHECK_DELAY:
                self.get_logger().info(f'等待方向信息更新，剩余时间: {self.DIRECTION_CHECK_DELAY - elapsed_time:.2f}秒')
                left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)
                return
            else:
                self.waiting_for_direction = False
                if self.current_crossroad_id in [1, 2] and self.state in [self.STATES['TO_WARD'], self.STATES['RETURNING']]:
                    self.forcing_straight = True
                    self.straight_start_time = time.time()
                    self.get_logger().info(f'十字路口 {self.current_crossroad_id} 不需要转弯，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                elif self.current_crossroad_id == 3:
                    if self.state == self.STATES['TO_WARD']:
                        directions = self.ward_directions.get(3, {})
                        ward_list = list(directions.keys())
                        if len(ward_list) > 0:
                            if len(ward_list) >= 2:
                                dir1 = directions[ward_list[0]]
                                dir2 = directions[ward_list[1]]
                                if dir1 == dir2:  # 同一侧
                                    self.turn_direction = dir1
                                    self.get_logger().info(f'病房 {ward_list[0]} 和 {ward_list[1]} 在同一侧，转向: {self.turn_direction}')
                                else:  # 两侧
                                    self.turn_direction = directions[self.target_wards[0]]
                                    self.get_logger().info(f'病房 {ward_list[0]} 和 {ward_list[1]} 在两侧，优先前往 {self.target_wards[0]}，转向: {self.turn_direction}')
                            else:
                                self.turn_direction = directions[ward_list[0]]
                            self.waiting_at_crossroad = True
                            self.enter_crossroad_time = time.time()
                            self.pid = PIDController(self.KP, self.KI, self.KD)
                            self.turn_history.append(self.turn_direction)
                            self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                        else:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'十字路口 3 无方向信息，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                    elif self.state == self.STATES['TO_SECOND_WARD']:
                        if self.wards_same_side:
                            directions = self.ward_directions.get(3, {})
                            if self.target_wards[1] in directions:
                                self.turn_direction = directions[self.target_wards[1]]
                                self.get_logger().info(f'前往第二个病房 {self.target_wards[1]} 时，十字路口 3 需要转向，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                            else:
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'前往第二个病房时，十字路口 3 无方向信息，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                        else:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'前往第二个病房时，十字路口 3 不需要转弯，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                    elif self.state == self.STATES['RETURNING']:
                        if 3 in self.ward_directions:
                            directions = self.ward_directions[3]
                            if directions[self.target_wards[0]] == directions[self.target_wards[1]]:
                                self.turn_direction = -self.turn_history[0]
                                self.get_logger().info(f'返回路径十字路口 3，病房在同一侧，转向: {self.turn_direction}')
                            else:
                                self.turn_direction = self.turn_history[0]
                                self.get_logger().info(f'返回路径十字路口 3，病房在两侧，转向: {self.turn_direction}')
                            self.waiting_at_crossroad = True
                            self.enter_crossroad_time = time.time()
                            self.pid = PIDController(self.KP, self.KI, self.KD)
                        else:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'返回路径十字路口 3 无方向信息，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                elif self.current_crossroad_id == 4:
                    if self.state == self.STATES['TO_WARD']:
                        directions = self.ward_directions.get(4, {})
                        if self.target_wards[0] in directions:
                            self.turn_direction = directions[self.target_wards[0]]
                            self.get_logger().info(f'右4路口，前往病房 {self.target_wards[0]}，转向: {self.turn_direction}')
                            self.waiting_at_crossroad = True
                            self.enter_crossroad_time = time.time()
                            self.pid = PIDController(self.KP, self.KI, self.KD)
                            self.turn_history.append(self.turn_direction)
                            self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                        else:
                            self.get_logger().warn(f'右4路口，病房 {self.target_wards[0]} 的方向信息缺失')
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'十字路口 4 无有效方向信息，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                    elif self.state == self.STATES['TO_SECOND_WARD']:
                        if not self.crossroad_4_passed:
                            self.get_logger().info(f'左4路口，crossroad_4_passed: {self.crossroad_4_passed}, turn_history: {self.turn_history}')
                            if not self.turn_history:
                                self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'左4路口，turn_history 为空，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = -self.turn_history[0]
                                self.get_logger().info(f'左4路口，反转进入病房 {self.target_wards[0]} 的方向，转向: {self.turn_direction}')
                                self.crossroad_4_passed = True
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                        else:
                            self.get_logger().info(f'右4路口，crossroad_4_passed: {self.crossroad_4_passed}, turn_history: {self.turn_history}')
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'右4路口，前往第二病房，暂无方向信息，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                    elif self.state == self.STATES['RETURNING']:
                        self.get_logger().info(f'返回路径右4路口，turn_history: {self.turn_history}')
                        self.turn_direction = -self.turn_history[-1]
                        self.get_logger().info(f'返回路径右4路口，反转进入病房 {self.target_wards[1]} 的方向，转向: {self.turn_direction}')
                        self.waiting_at_crossroad = True
                        self.enter_crossroad_time = time.time()
                        self.pid = PIDController(self.KP, self.KI, self.KD)
                elif self.current_crossroad_id == 5:
                    if self.state == self.STATES['TO_SECOND_WARD']:
                        if self.wards_same_side:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'第五个路口，病房在同一侧，直行前往第二个病房，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                        else:
                            if not self.turn_history:
                                self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'第五个路口，turn_history 为空，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = -self.turn_history[1]
                                self.get_logger().info(f'第五个路口，复用第四个路口的方向反转，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                elif self.current_crossroad_id == 6:
                    if self.state == self.STATES['RETURNING']:
                        if self.wards_same_side:
                            if len(self.turn_history) < 2:
                                self.get_logger().error('turn_history 不足，无法复用路口 4 方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'第六个路口，turn_history 不足，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = self.turn_history[1]  # 复用路口 4 方向
                                self.get_logger().info(f'第六个路口，复用第四个路口的方向，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                        else:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'十字路口 {self.current_crossroad_id} 跳过病房检测，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                elif self.current_crossroad_id == 7:
                    if self.state == self.STATES['TO_SECOND_WARD']:
                        directions = self.ward_directions.get(7, {})
                        if self.target_wards[1] in directions:
                            self.turn_direction = directions[self.target_wards[1]]
                            self.get_logger().info(f'第七个路口，前往病房 {self.target_wards[1]}，转向: {self.turn_direction}')
                        else:
                            self.get_logger().warn(f'第七个路口，病房 {self.target_wards[1]} 的方向信息缺失，使用病房 {self.target_wards[0]} 的方向反转')
                            self.turn_direction = -directions[self.target_wards[0]]
                        self.waiting_at_crossroad = True
                        self.enter_crossroad_time = time.time()
                        self.pid = PIDController(self.KP, self.KI, self.KD)
                        self.turn_history.append(self.turn_direction)
                        self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                    elif self.state == self.STATES['RETURNING']:
                        if self.wards_same_side:
                            if not self.turn_history:
                                self.get_logger().error('turn_history 为空，无法反转路口 3 方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'第七个路口，turn_history 为空，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = -self.turn_history[0]  # 反转路口 3 方向
                                self.get_logger().info(f'第七个路口，反转第三个路口的方向，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                        else:
                            self.turn_direction = -self.turn_history[-1]
                            self.get_logger().info(f'返回路径第七个路口，反转进入病房 {self.target_wards[1]} 的方向，转向: {self.turn_direction}')
                            self.waiting_at_crossroad = True
                            self.enter_crossroad_time = time.time()
                            self.pid = PIDController(self.KP, self.KI, self.KD)
                elif self.current_crossroad_id == 8:
                    if self.state == self.STATES['RETURNING']:
                        if self.wards_same_side:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'第八个路口，病房在同一侧，直行返回药房，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                        else:
                            if not self.turn_history:
                                self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'第八个路口，turn_history 为空，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = -self.turn_history[-1]
                                self.get_logger().info(f'第八个路口，复用第七个路口的方向反转，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                elif self.current_crossroad_id == 9:
                    if self.state == self.STATES['RETURNING']:
                        if self.wards_same_side:
                            self.forcing_straight = True
                            self.straight_start_time = time.time()
                            self.get_logger().info(f'第九个路口，病房在同一侧，直行返回药房，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                        else:
                            if not self.turn_history:
                                self.get_logger().error('turn_history 为空，无法复用方向，强制直行')
                                self.forcing_straight = True
                                self.straight_start_time = time.time()
                                self.get_logger().info(f'第九个路口，turn_history 为空，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')
                            else:
                                self.turn_direction = self.turn_history[0]
                                self.get_logger().info(f'第九个路口，复用第三个路口的方向，转向: {self.turn_direction}')
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                self.turn_history.append(self.turn_direction)
                                self.crossroad_history.append((self.current_crossroad_id, self.turn_direction))
                elif self.current_crossroad_id in [10, 11]:
                    if self.state == self.STATES['RETURNING']:
                        self.forcing_straight = True
                        self.straight_start_time = time.time()
                        self.get_logger().info(f'十字路口 {self.current_crossroad_id} 直行，强制直行 {self.STRAIGHT_WAIT_TIME} 秒')

        if self.forcing_straight:
            elapsed_time = time.time() - self.straight_start_time
            if elapsed_time < self.STRAIGHT_WAIT_TIME:
                self.get_logger().info(f'强制直行中，剩余时间: {self.STRAIGHT_WAIT_TIME - elapsed_time:.2f}秒')
                left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)
                return
            else:
                self.get_logger().info('强制直行结束，恢复巡线')
                self.forcing_straight = False

        if self.state == self.STATES['TO_WARD']:
            if self.waiting_at_crossroad:
                elapsed_time = time.time() - self.enter_crossroad_time
                if elapsed_time < self.WAIT_AT_CROSSROAD_TIME:
                    self.get_logger().info(f'等待到达十字路口中心，剩余时间: {self.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                    left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                else:
                    self.get_logger().info('已到达十字路口中心，开始转向')
                    self.waiting_at_crossroad = False
                    self.turning = True
                    self.entered_crossroad = False
                    self.line_left_center = False
                    self.perform_turn(line_center)
            elif self.turning:
                self.perform_turn(line_center)
            else:
                if line_center is not None:
                    self.line_lost_count = 0
                    error = line_center - self.image_center
                    pid_output = self.pid.compute(error)
                    left_speed = self.BASE_SPEED + pid_output
                    right_speed = self.BASE_SPEED - pid_output
                    left_speed = max(min(left_speed, self.MAX_SPEED), -self.MAX_SPEED)
                    right_speed = max(min(right_speed, self.MAX_SPEED), -self.MAX_SPEED)
                else:
                    self.line_lost_count += 1
                    if self.line_lost_count > self.MAX_LINE_LOST:
                        self.get_logger().info(f'到达病房 {self.target_wards[self.current_ward_index]}')
                        self.state = self.STATES['UNLOADING']
                        self.unload_start_time = time.time()
                        self.line_lost_count = 0
                        self.publish_cmd_vel(0.0, 0.0)
                        return
                    left_speed = right_speed = self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)

        elif self.state == self.STATES['TO_SECOND_WARD']:
            if self.waiting_at_crossroad:
                elapsed_time = time.time() - self.enter_crossroad_time
                if elapsed_time < self.WAIT_AT_CROSSROAD_TIME:
                    self.get_logger().info(f'等待到达十字路口中心，剩余时间: {self.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                    left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                else:
                    self.get_logger().info('已到达十字路口中心，开始转向')
                    self.waiting_at_crossroad = False
                    self.turning = True
                    self.entered_crossroad = False
                    self.line_left_center = False
                    self.perform_turn(line_center)
            elif self.turning:
                self.perform_turn(line_center)
            else:
                if line_center is not None:
                    self.line_lost_count = 0
                    error = line_center - self.image_center
                    pid_output = self.pid.compute(error)
                    left_speed = self.BASE_SPEED + pid_output
                    right_speed = self.BASE_SPEED - pid_output
                    left_speed = max(min(left_speed, self.MAX_SPEED), -self.MAX_SPEED)
                    right_speed = max(min(right_speed, self.MAX_SPEED), -self.MAX_SPEED)
                else:
                    self.line_lost_count += 1
                    if self.line_lost_count > self.MAX_LINE_LOST:
                        self.get_logger().info(f'到达病房 {self.target_wards[self.current_ward_index]}')
                        self.state = self.STATES['UNLOADING']
                        self.unload_start_time = time.time()
                        self.line_lost_count = 0
                        self.publish_cmd_vel(0.0, 0.0)
                        return
                    left_speed = right_speed = self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)

        elif self.state == self.STATES['UNLOADING']:
            if time.time() - self.unload_start_time >= self.UNLOAD_TIME:
                self.get_logger().info('卸载药品完成')
                self.state = self.STATES['TURNING_AROUND']
                self.turn_around_start_time = time.time()
                self.get_logger().info('开始180°掉头')
            self.publish_cmd_vel(0.0, 0.0)

        elif self.state == self.STATES['TURNING_AROUND']:
            if time.time() - self.turn_around_start_time >= self.TURN_AROUND_TIME:
                self.get_logger().info('掉头完成')
                self.current_ward_index += 1
                if self.current_ward_index < len(self.target_wards):
                    self.state = self.STATES['TO_SECOND_WARD']
                    self.last_crossroad_id = 0
                    self.crossroad_4_passed = False
                    self.get_logger().info(f'前往下一个病房: {self.target_wards[self.current_ward_index]}')
                    self.get_logger().info(f'进入 TO_SECOND_WARD 状态，crossroad_4_passed: {self.crossroad_4_passed}')
                else:
                    self.state = self.STATES['RETURNING']
                    self.return_turn_index = len(self.turn_history) - 1
                    self.last_crossroad_id = 0
                    self.get_logger().info(f'开始返回药房，turn_history: {self.turn_history}')
                self.pid = PIDController(self.KP, self.KI, self.KD)
            else:
                self.perform_turn_around()

        elif self.state == self.STATES['RETURNING']:
            if self.waiting_at_crossroad:
                elapsed_time = time.time() - self.enter_crossroad_time
                if elapsed_time < self.WAIT_AT_CROSSROAD_TIME:
                    self.get_logger().info(f'等待到达十字路口中心，剩余时间: {self.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                    left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                else:
                    self.get_logger().info('已到达十字路口中心，开始转向')
                    self.waiting_at_crossroad = False
                    self.turning = True
                    self.entered_crossroad = False
                    self.line_left_center = False
                    self.perform_turn(line_center)
            elif self.turning:
                self.perform_turn(line_center)
            else:
                if line_center is not None:
                    self.line_lost_count = 0
                    error = line_center - self.image_center
                    pid_output = self.pid.compute(error)
                    left_speed = self.BASE_SPEED + pid_output
                    right_speed = self.BASE_SPEED - pid_output
                    left_speed = max(min(left_speed, self.MAX_SPEED), -self.MAX_SPEED)
                    right_speed = max(min(right_speed, self.MAX_SPEED), -self.MAX_SPEED)
                else:
                    self.line_lost_count += 1
                    if self.line_lost_count > self.MAX_LINE_LOST:
                        self.get_logger().info(f'返回药房，总用时: {time.time() - self.start_time:.2f}s')
                        self.state = self.STATES['WAITING']
                        self.current_ward_index = 0
                        self.turn_history = []
                        self.crossroad_history = []
                        self.return_turn_index = 0
                        self.current_crossroad_id = 0
                        self.last_crossroad_id = 0
                        self.get_logger().info('到达药房，停车')
                        self.publish_cmd_vel(0.0, 0.0)
                        self.mode = 1
                        return
                    left_speed = right_speed = self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)

        frame = self.draw_debug_info(frame, line_center, center_y, left_speed, right_speed, self.state, bounding_box, self.turning)
        binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(binary_display, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def perform_turn(self, line_center):
        left_speed = right_speed = 0.0
        if not self.entered_crossroad:
            self.entered_crossroad = True
            self.get_logger().info('开始转向')
            if self.turn_direction == -1:
                left_speed, right_speed = -20.0, 20.0
                self.get_logger().info('正在左转')
            else:
                left_speed, right_speed = 20.0, -20.0
                self.get_logger().info('正在右转')
            self.publish_cmd_vel(left_speed, right_speed)
        else:
            if self.turn_direction == -1:
                left_speed, right_speed = -20.0, 20.0
                self.get_logger().info('正在左转')
            else:
                left_speed, right_speed = 20.0, -20.0
                self.get_logger().info('正在右转')
            self.publish_cmd_vel(left_speed, right_speed)

            if not self.line_left_center and line_center is not None:
                error = abs(line_center - self.image_center)
                if error > 300:
                    self.line_left_center = True
                    self.get_logger().info(f'红线已离开中心，line_center={line_center}，image_center={self.image_center}，误差={error}')

            if self.line_left_center and line_center is not None:
                error = abs(line_center - self.image_center)
                if error < self.CENTER_THRESHOLD:
                    self.get_logger().info(f'红线回到中间，line_center={line_center}，image_center={self.image_center}，误差={error}')
                    self.turning = False
                    self.turn_direction = 0
                    self.line_left_center = False
                    self.get_logger().info('转向完成')
                    left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                    self.publish_cmd_vel(left_speed, right_speed)

    def perform_turn_around(self):
        left_speed, right_speed = -50.0, 50.0
        self.get_logger().info('正在180°掉头')
        self.publish_cmd_vel(left_speed, right_speed)

    def publish_cmd_vel(self, left_speed, right_speed):
        msg = Float32MultiArray()
        msg.data = [float(left_speed), float(right_speed), float(self.mode)]
        self.cmd_vel_pub.publish(msg)

    def draw_debug_info(self, frame, line_center, center_y, left_speed, right_speed, state, bounding_box=None, is_turning=False):
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        cv2.line(frame_copy, (self.image_center, 0), (self.image_center, h), (0, 255, 0), 1)

        fixed_y_line = int(h * 0.875)
        cv2.line(frame_copy, (0, fixed_y_line), (w, fixed_y_line), (255, 255, 0), 1)

        fixed_y_turn = int(h * 0.25)
        cv2.line(frame_copy, (0, fixed_y_turn), (w, fixed_y_turn), (255, 0, 255), 1)

        if line_center is not None and center_y is not None:
            try:
                line_center = int(line_center)
                center_y = int(center_y)
                if 0 <= center_y < h and 0 <= line_center < w:
                    color = (0, 0, 255) if not is_turning else (255, 0, 0)
                    cv2.circle(frame_copy, (line_center, center_y), 5, color, -1)
                    error = line_center - self.image_center
                    cv2.putText(frame_copy, f"误差: {error:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except (TypeError, ValueError) as e:
                self.get_logger().error(f'绘制偏移点失败: {str(e)}, line_center={line_center}, center_y={center_y}')
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame_copy, f"左轮: {left_speed:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_copy, f"右轮: {right_speed:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_copy, f"状态: {state}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_copy, f"目标病房: {self.target_wards}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"当前病房索引: {self.current_ward_index}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return frame_copy

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()