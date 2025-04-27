import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from std_msgs.msg import Float32MultiArray, Int8MultiArray, Float32
from .one_number import SingleWardMode
from .double_number import DualWardMode

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
    print(f"Image shape: {image.shape}")  # 添加日志
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

    roi_upper = image[0:int(h*0.2), :, :]
    hsv_upper = cv2.cvtColor(roi_upper, cv2.COLOR_BGR2HSV)
    mask1_upper = cv2.inRange(hsv_upper, lower_red1, upper_red1)
    mask2_upper = cv2.inRange(hsv_upper, lower_red2, upper_red2)
    mask_upper = mask1_upper | mask2_upper
    mask_upper = cv2.morphologyEx(mask_upper, cv2.MORPH_OPEN, kernel)
    contours_upper, _ = cv2.findContours(mask_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    start_col = int(w * 0.25)
    end_col = int(w * 0.75)
    roi_left = image[start_col:end_col, 0:int(w*0.2), :]
    hsv_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2HSV)
    mask1_left = cv2.inRange(hsv_left, lower_red1, upper_red1)
    mask2_left = cv2.inRange(hsv_left, lower_red2, upper_red2)
    mask_left = mask1_left | mask2_left
    mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_OPEN, kernel)
    contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_right = image[:, int(w*0.8):w, :]  # 修复索引
    hsv_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2HSV)
    mask1_right = cv2.inRange(hsv_right, lower_red1, upper_red1)
    mask2_right = cv2.inRange(hsv_right, lower_red2, upper_red2)
    mask_right = mask1_right | mask2_right
    mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_OPEN, kernel)
    contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return line_center, center_y, contours_line, current_mask, bounding_box, contours_upper, contours_left, contours_right

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
        self.MAX_LINE_LOST = 2  # 增加以降低误触发
        self.UNLOAD_TIME =1.0
        self.TURN_TIME = 0.85
        self.TURN_AROUND_TIME = 1.5
        self.CENTER_THRESHOLD = 35
        self.WAIT_AT_CROSSROAD_TIME = 0.1
        self.STRAIGHT_WAIT_TIME = 0.5
        self.QUALITY_THRESHOLD_FIRST = 120.0  # 第一个病房卸货后质量阈值
        self.QUALITY_THRESHOLD_SECOND = 20.0   # 第二个病房卸货后质量阈值
        self.INITIAL_QUALITY = 180.0          # 调整为实际初始质量
        self.UNLOAD_TIMEOUT = 10.0            # 卸货超时时间（秒）

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
        self.quality = 0.0  # 初始化质量
        self.target_wards = []
        self.crossroad_history = []
        self.current_crossroad_id = 0
        self.current_ward_index = 0
        self.turn_history = []
        self.return_turn_index = 0
        self.crossroad_count = 0
        self.start_time = None
        self.unload_start_time = None
        self.turn_around_start_time = None
        self.enter_crossroad_time = None
        self.turning = False
        self.waiting_at_crossroad = False
        self.entered_crossroad = False
        self.line_left_center = False
        self.line_lost_count = 0
        self.turn_direction = 0
        self.lock_turn_direction = False
        self.just_entered_unloading = False  # 跟踪是否刚进入UNLOADING
        self.image_center = self.IMAGE_WIDTH // 2
        self.pid = PIDController(self.KP, self.KI, self.KD)

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.direction_sub = self.create_subscription(Int8MultiArray, '/direction', self.get_direction, 10)
        self.mode_sub = self.create_subscription(Float32, '/serial_mode', self.update_mode, 10)
        self.ward_sub = self.create_subscription(Int8MultiArray, '/target_wards', self.update_target_wards, 10)
        self.cmd_vel_pub = self.create_publisher(Float32MultiArray, '/cmd_vel', 10)

        self.single_ward_mode = SingleWardMode(self)
        self.dual_ward_mode = DualWardMode(self)

        self.get_logger().info('线跟踪节点已启动')

    def detect_crossroad(self, contours_upper, contours_left, contours_right, image_width):
        def has_significant_contour(contours, min_area=10, min_width_ratio=0.05):
            if not contours:
                return False
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if area > min_area and w > image_width * min_width_ratio:
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

        if is_crossroad:
            self.get_logger().info("检测到十字路口")
            return True
        else:
            self.get_logger().debug("未检测到十字路口")
            return False

    def get_direction(self, msg):
        if self.lock_turn_direction:
            self.get_logger().info('turn_direction 已锁定，忽略 /direction 消息')
            return
        if self.state in [self.STATES['UNLOADING'], self.STATES['TURNING_AROUND'], self.STATES['RETURNING']]:
            return
        self.turn_direction = msg.data[0] if msg.data[0] in [-1, 1] else 0

    def update_mode(self, msg):
        self.quality = msg.data
        self.get_logger().info(f'更新质量: {self.quality}')
        if self.state == self.STATES['WAITING']:
            self.mode = 2 if self.quality >= self.INITIAL_QUALITY else 1
            if self.mode == 2:
                if not self.target_wards:
                    self.get_logger().warn('目标病房未设置，等待 /target_wards')
                    self.mode = 1
                    return
                self.state = self.STATES['TO_WARD']
                self.start_time = time.time()
                self.crossroad_count = 0
                self.get_logger().info(f'开始送药，目标病房: {self.target_wards}')
        elif self.state == self.STATES['RETURNING']:
            self.mode = 1
            self.state = self.STATES['WAITING']
            self.crossroad_count = 0
            self.get_logger().info('返回药房完成')

    def update_target_wards(self, msg):
        self.target_wards = list(msg.data)
        self.get_logger().info(f'更新目标病房: {self.target_wards}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {str(e)}')
            return

        line_center, center_y, contours_line, binary, bounding_box, contours_upper, contours_left, contours_right = process_image(
            frame, is_turning=self.turning, turn_direction=self.turn_direction
        )
        left_speed = right_speed = 0.0

        if self.state == self.STATES['WAITING']:
            self.publish_cmd_vel(0.0, 0.0)
            return

        elif self.state == self.STATES['TO_WARD']:
            left_speed, right_speed = self.single_ward_mode.handle(
                line_center, contours_upper, contours_left, contours_right
            )

        elif self.state == self.STATES['TO_SECOND_WARD']:
            left_speed, right_speed = self.dual_ward_mode.handle(
                line_center, contours_upper, contours_left, contours_right
            )

        elif self.state == self.STATES['UNLOADING']:
            if len(self.target_wards) == 1:  # 单病房模式
                if time.time() - self.unload_start_time >= self.UNLOAD_TIME:
                    self.get_logger().info(f'病房 {self.target_wards[self.current_ward_index]} 卸货完成（等待 {self.UNLOAD_TIME}秒）')
                    self.state = self.STATES['TURNING_AROUND']
                    self.turn_around_start_time = time.time()
                    self.get_logger().info('开始180°掉头，返回药房')
                else:
                    self.get_logger().info(f'正在卸货，剩余时间: {self.UNLOAD_TIME - (time.time() - self.unload_start_time):.2f}秒')
            elif len(self.target_wards) == 2:  # 双病房模式
                self.get_logger().debug(f'UNLOADING: just_entered_unloading={self.just_entered_unloading}')  # 添加日志
                if self.just_entered_unloading:  # 刚进入UNLOADING状态
                    self.mode = 1
                    self.just_entered_unloading = False
                    self.get_logger().info(f'到达病房 {self.target_wards[self.current_ward_index]}，切换 mode 为 1（卸货中）')
                self.mode = 1
                quality_threshold = self.QUALITY_THRESHOLD_FIRST if self.current_ward_index == 0 else self.QUALITY_THRESHOLD_SECOND
                if self.quality <= quality_threshold:
                    self.get_logger().info(f'病房 {self.target_wards[self.current_ward_index]} 卸货完成，当前质量: {self.quality}')
                    if self.current_ward_index == 0:  # 第一个病房
                        self.state = self.STATES['TURNING_AROUND']  # 先掉头
                        self.mode = 2
                        self.turn_around_start_time = time.time()
                        self.get_logger().info('第一个病房卸货完成，开始180°掉头')
                    else:  # 第二个病房
                        self.state = self.STATES['TURNING_AROUND']
                        self.mode = 2
                        self.turn_around_start_time = time.time()
                        self.get_logger().info('第二个病房卸货完成，开始180°掉头，返回药房')
                elif time.time() - self.unload_start_time >= self.UNLOAD_TIMEOUT:  # 卸货超时
                    self.get_logger().warn(f'卸货超时，当前质量: {self.quality}，强制继续')
                    if self.current_ward_index == 0:
                        self.state = self.STATES['TURNING_AROUND']
                        self.mode = 2
                        self.turn_around_start_time = time.time()
                        self.get_logger().info('第一个病房卸货超时，开始180°掉头')
                    else:
                        self.state = self.STATES['TURNING_AROUND']
                        self.mode = 2
                        self.turn_around_start_time = time.time()
                        self.get_logger().info('第二个病房卸货超时，开始180°掉头，返回药房')
                else:
                    self.get_logger().info(f'正在卸货，当前质量: {self.quality}，等待质量 <= {quality_threshold}')
            self.publish_cmd_vel(0.0, 0.0)

        elif self.state == self.STATES['TURNING_AROUND']:
            if time.time() - self.turn_around_start_time >= self.TURN_AROUND_TIME:
                self.get_logger().info('掉头完成')
                if len(self.target_wards) == 2 and self.current_ward_index == 0:  # 第一个病房掉头后
                    self.current_ward_index += 1
                    self.state = self.STATES['TO_SECOND_WARD']
                    self.mode = 2
                    self.crossroad_count = 0
                    self.get_logger().info(f'准备前往第二个病房: {self.target_wards[self.current_ward_index]}，切换 mode 为 2（出发），从十字路口 {self.current_crossroad_id} 开始')
                    self.pid = PIDController(self.KP, self.KI, self.KD)
                else:  # 第二个病房或单病房掉头后
                    self.state = self.STATES['RETURNING']
                    self.return_turn_index = len(self.turn_history) - 1
                    self.crossroad_count = 0
                    self.get_logger().info(f'开始返回药房，turn_history: {self.turn_history}，crossroad_history: {self.crossroad_history}')
                    self.pid = PIDController(self.KP, self.KI, self.KD)
            else:
                self.perform_turn_around()

        elif self.state == self.STATES['RETURNING']:
            if not self.turning and not self.waiting_at_crossroad and self.detect_crossroad(contours_upper, contours_left, contours_right, self.IMAGE_WIDTH):
                self.crossroad_count += 1
                if self.crossroad_count == 1:
                    self.current_crossroad_id = 4
                elif self.crossroad_count == 2:
                    self.current_crossroad_id = 3
                elif self.crossroad_count == 3:
                    self.current_crossroad_id = 2
                else:
                    self.current_crossroad_id = 1

                self.get_logger().info(f'返回路径检测到十字路口 {self.current_crossroad_id} (计数: {self.crossroad_count}/{len(self.turn_history)})')

                if self.crossroad_count <= len(self.turn_history):
                    if self.current_crossroad_id in [1, 2]:
                        self.turn_direction = 0
                        self.get_logger().info(f'返回路径十字路口 {self.current_crossroad_id}，强制直行')
                        self.publish_cmd_vel(self.BASE_SPEED, self.BASE_SPEED)
                        time.sleep(self.STRAIGHT_WAIT_TIME)
                    else:
                        reverse_index = len(self.turn_history) - self.crossroad_count
                        if reverse_index >= 0:
                            reverse_turn = -self.turn_history[reverse_index]
                            self.get_logger().info(f'返回路径十字路口 {self.current_crossroad_id}，转向: {reverse_turn} (索引: {reverse_index}/{len(self.turn_history)-1})')
                            if reverse_turn != 0:
                                self.turn_direction = reverse_turn
                                self.waiting_at_crossroad = True
                                self.enter_crossroad_time = time.time()
                                self.pid = PIDController(self.KP, self.KI, self.KD)
                                left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                            else:
                                self.get_logger().info(f'返回路径十字路口 {self.current_crossroad_id} 无转向记录，直行')
                                self.publish_cmd_vel(self.BASE_SPEED, self.BASE_SPEED)
                                time.sleep(self.STRAIGHT_WAIT_TIME)
                else:
                    self.get_logger().info(f'返回路径十字路口计数超出 turn_history 长度，直行')
                    self.publish_cmd_vel(self.BASE_SPEED, self.BASE_SPEED)
                    time.sleep(self.STRAIGHT_WAIT_TIME)

            elif self.waiting_at_crossroad:
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
                        if self.crossroad_count >= len(self.turn_history):
                            self.get_logger().info(f'返回药房，总用时: {time.time() - self.start_time:.2f}s')
                            self.state = self.STATES['WAITING']
                            self.current_ward_index = 0
                            self.turn_history = []
                            self.crossroad_history = []
                            self.return_turn_index = 0
                            self.crossroad_count = 0
                            self.current_crossroad_id = 0
                            self.publish_cmd_vel(0.0, 0.0)
                            self.mode = 1
                            return
                        else:
                            self.get_logger().warn(f'线丢失但未经过所有十字路口，继续前行 (十字路口计数: {self.crossroad_count}/{len(self.turn_history)})')
                            left_speed = right_speed = self.BASE_SPEED
                    else:
                        left_speed = right_speed = self.BASE_SPEED
                self.publish_cmd_vel(left_speed, right_speed)

        frame = self.draw_debug_info(frame, line_center, center_y, left_speed, right_speed, self.state, bounding_box, self.turning)
        binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(binary_display, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

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

        upper_top_left = (0, 0)
        upper_bottom_right = (w, int(h*0.2))
        cv2.rectangle(frame_copy, upper_top_left, upper_bottom_right, (0, 255, 255), 2)
        cv2.putText(frame_copy, "上区域", (10, int(h*0.2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        left_top_left = (0, 0)
        left_bottom_right = (int(w*0.2), h)
        cv2.rectangle(frame_copy, left_top_left, left_bottom_right, (255, 0, 0), 2)
        cv2.putText(frame_copy, "左区域", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        right_top_left = (int(w*0.8), 0)
        right_bottom_right = (w, h)
        cv2.rectangle(frame_copy, right_top_left, right_bottom_right, (0, 255, 0), 2)
        cv2.putText(frame_copy, "右区域", (int(w*0.8) + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if line_center is not None and center_y is not None:
            try:
                line_center = int(line_center)
                center_y = int(center_y)
                if 0 <= center_y < h and 0 <= line_center < w:
                    color = (0, 0, 255) if not is_turning else (255, 0, 0)
                    cv2.circle(frame_copy, (line_center, center_y), 5, color, -1)
                    error = line_center - self.image_center
                    cv2.putText(frame_copy, f"误差: {error:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.get_logger().warn(f'偏移点坐标超出图像范围: (x={line_center}, y={center_y})')
            except (TypeError, ValueError) as e:
                self.get_logger().error(f'绘制偏移点失败: {str(e)}, line_center={line_center}, center_y={center_y}')
        if bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame_copy, f"左轮: {left_speed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_copy, f"右轮: {right_speed:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_copy, f"状态: {state}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_copy, f"目标病房: {self.target_wards}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"十字路口计数: {self.crossroad_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_copy, f"当前质量: {self.quality:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame_copy, f"模式: {self.mode}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        return frame_copy

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点已停止')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()