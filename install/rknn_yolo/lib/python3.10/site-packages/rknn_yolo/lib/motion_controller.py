import time
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int8

class MotionController:
    def __init__(self, node):
        self.node = node
        self.IMAGE_WIDTH = 640
        self.BASE_SPEED = 90.0
        self.MAX_SPEED = 120.0
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
        self.ward_directions = {}
        self.crossroad_4_passed = False
        self.wards_same_side = False

    def crossroad_callback(self, msg):
        if self.turning or self.waiting_at_crossroad:
            self.node.get_logger().info(f'当前处于转弯或等待路口中心状态，忽略十字路口ID更新: {msg.data}')
            return

        previous_crossroad_id = self.current_crossroad_id
        self.current_crossroad_id = msg.data
        if self.current_crossroad_id != self.last_crossroad_id:
            self.last_crossroad_id = self.current_crossroad_id
            self.waiting_for_direction = True
            self.direction_check_start_time = time.time()
            self.node.get_logger().info(f'收到十字路口ID: {self.current_crossroad_id}（前一个ID: {previous_crossroad_id}）')

    def direction_callback(self, msg):
        crossroad_id = msg.data[0]
        num_wards = msg.data[1]
        ward_directions = {}
        for i in range(num_wards):
            ward = msg.data[2 + i*2]
            direction = msg.data[3 + i*2]
            ward_directions[ward] = direction
        self.ward_directions[crossroad_id] = ward_directions
        self.node.get_logger().info(f'收到病房方向信息: 十字路口 {crossroad_id}, 方向: {self.ward_directions}')

        if crossroad_id == 3 and len(self.target_wards) >= 2 and not self.wards_same_side:
            ward1, ward2 = self.target_wards[0], self.target_wards[1]
            if ward1 in ward_directions and ward2 in ward_directions:
                self.wards_same_side = (ward_directions[ward1] == ward_directions[ward2])
                self.node.get_logger().info(f'路口 3 病房方向判断: 病房 {ward1} 方向 {ward_directions[ward1]}, 病房 {ward2} 方向 {ward_directions[ward2]}, 同一侧: {self.wards_same_side}')

    def update_mode(self, msg):
        self.quality = msg.data
        if self.state == self.STATES['WAITING']:
            self.mode = 2 if self.quality >= 200 else 1
            if self.mode == 2:
                if not self.target_wards:
                    self.node.get_logger().warn('目标病房未设置，等待 /target_wards')
                    self.mode = 1
                    return
                self.state = self.STATES['TO_WARD']
                self.start_time = time.time()
                self.last_crossroad_id = 0
                self.current_ward_index = 0
                self.node.get_logger().info(f'开始送药，目标病房: {self.target_wards}')
        elif self.state == self.STATES['RETURNING']:
            self.mode = 1
            self.state = self.STATES['WAITING']
            self.last_crossroad_id = 0
            self.current_ward_index = 0
            self.node.get_logger().info('返回药房完成')

    def update_target_wards(self, msg):
        self.target_wards = list(msg.data)
        self.node.get_logger().info(f'更新目标病房: {self.target_wards}')
        if self.state == self.STATES['WAITING'] and self.mode == 2:
            self.state = self.STATES['TO_WARD']
            self.start_time = time.time()
            self.last_crossroad_id = 0
            self.current_ward_index = 0
            self.node.get_logger().info(f'开始送药，目标病房: {self.target_wards}')

    def perform_turn(self, line_center, pid_controller):
        #left_speed = right_speed = 0.0
        if not self.entered_crossroad:
            self.entered_crossroad = True
            self.node.get_logger().info('开始转向')
            if self.turn_direction == -1:
                left_speed, right_speed = -20.0, 20.0
                self.node.get_logger().info('正在左转')
            else:
                left_speed, right_speed = 20.0, -20.0
                self.node.get_logger().info('正在右转')
            self.publish_cmd_vel(left_speed, right_speed)
        else:
            if self.turn_direction == -1:
                left_speed, right_speed = -20.0, 20.0
                self.node.get_logger().info('正在左转')
            else:
                left_speed, right_speed = 20.0, -20.0
                self.node.get_logger().info('正在右转')
            self.publish_cmd_vel(left_speed, right_speed)

            if not self.line_left_center and line_center is not None:
                error = abs(line_center - self.image_center)
                if error > 300:
                    self.line_left_center = True
                    self.node.get_logger().info(f'红线已离开中心，line_center={line_center}，image_center={self.image_center}，误差={error}')

            if self.line_left_center and line_center is not None:
                error = abs(line_center - self.image_center)
                if error < self.CENTER_THRESHOLD:
                    self.node.get_logger().info(f'红线回到中间，line_center={line_center}，image_center={self.image_center}，误差={error}')
                    self.turning = False
                    self.turn_direction = 0
                    self.line_left_center = False
                    self.node.get_logger().info('转向完成')
                    left_speed, right_speed = self.BASE_SPEED, self.BASE_SPEED
                    self.publish_cmd_vel(left_speed, right_speed)

    def perform_turn_around(self):
        left_speed, right_speed = -50.0, 50.0
        self.node.get_logger().info('正在180°掉头')
        self.publish_cmd_vel(left_speed, right_speed)

    def publish_cmd_vel(self, left_speed, right_speed):
        msg = Float32MultiArray()
        msg.data = [float(left_speed), float(right_speed), float(self.mode)]
        self.node.cmd_vel_pub.publish(msg)
    
    def publish_reset(self):
        msg = Int8()
        msg.data = 0
        self.node.reset.publish(msg)