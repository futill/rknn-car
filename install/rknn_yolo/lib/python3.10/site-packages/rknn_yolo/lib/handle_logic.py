import time

class handle_logic:
    def __init__(self, node):
        self.node = node  # 存储 LineFollowingNode 实例
        self.motion = node.motion  # 存储 motion_controller 实例
        self.pid = node.pid

    def get_logger(self):
        return self.node.get_logger()  # 使用节点的日志器
    
    def handle_crossroad_logic(self):
        self.motion.waiting_for_direction = False
        if self.motion.current_crossroad_id in [1, 2] and self.motion.state in [self.motion.STATES['TO_WARD'], self.motion.STATES['RETURNING']]:
            self.motion.forcing_straight = True
            self.motion.straight_start_time = time.time()
            self.get_logger().info(f'十字路口 {self.motion.current_crossroad_id} 不需要转弯，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
        elif self.motion.current_crossroad_id == 3:
            self.handle_crossroad_3()
        elif self.motion.current_crossroad_id == 4:
            self.handle_crossroad_4()
        elif self.motion.current_crossroad_id == 5:
            self.handle_crossroad_5()
        elif self.motion.current_crossroad_id == 6:
            self.handle_crossroad_6()
        elif self.motion.current_crossroad_id == 7:
            self.handle_crossroad_7()
        elif self.motion.current_crossroad_id == 8:
            self.handle_crossroad_8()
        elif self.motion.current_crossroad_id == 9:
            self.handle_crossroad_9()
        elif self.motion.current_crossroad_id in [10, 11]:
            self.handle_crossroad_10_11()

    def handle_crossroad_3(self):
        if self.motion.state == self.motion.STATES['TO_WARD']:
            directions = self.motion.ward_directions.get(3, {})
            ward_list = list(directions.keys())
            if len(ward_list) > 0:
                if len(ward_list) >= 2:
                    dir1 = directions[ward_list[0]]
                    dir2 = directions[ward_list[1]]
                    if dir1 == dir2:
                        self.motion.turn_direction = dir1
                        self.get_logger().info(f'病房 {ward_list[0]} 和 {ward_list[1]} 在同一侧，转向: {self.motion.turn_direction}')
                    else:
                        self.motion.turn_direction = directions[self.motion.target_wards[0]]
                        self.get_logger().info(f'病房 {ward_list[0]} 和 {ward_list[1]} 在两侧，优先前往 {self.motion.target_wards[0]}，转向: {self.motion.turn_direction}')
                else:
                    self.motion.turn_direction = directions[ward_list[0]]
                self.motion.waiting_at_crossroad = True
                self.motion.enter_crossroad_time = time.time()
                self.motion.turn_history.append(self.motion.turn_direction)
                self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
            else:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'十字路口 3 无方向信息，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
        elif self.motion.state == self.motion.STATES['TO_SECOND_WARD']:
            if self.motion.wards_same_side:
                directions = self.motion.ward_directions.get(3, {})
                if self.motion.target_wards[1] in directions:
                    self.motion.turn_direction = directions[self.motion.target_wards[1]]
                    self.get_logger().info(f'前往第二个病房 {self.motion.target_wards[1]} 时，十字路口 3 需要转向，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
                else:
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'前往第二个病房时，十字路口 3 无方向信息，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
            else:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'前往第二个病房时，十字路口 3 不需要转弯，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
        elif self.motion.state == self.motion.STATES['RETURNING']:
            if 3 in self.motion.ward_directions:
                directions = self.motion.ward_directions[3]
                if directions[self.motion.target_wards[0]] == directions[self.motion.target_wards[1]]:
                    self.motion.turn_direction = -self.motion.turn_history[0]
                    self.get_logger().info(f'返回路径十字路口 3，病房在同一侧，转向: {self.motion.turn_direction}')
                else:
                    self.motion.turn_direction = self.motion.turn_history[0]
                    self.get_logger().info(f'返回路径十字路口 3，病房在两侧，转向: {self.motion.turn_direction}')
                self.motion.waiting_at_crossroad = True
                self.motion.enter_crossroad_time = time.time()
            else:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'返回路径十字路口 3 无方向信息，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')

    def handle_crossroad_4(self):
        if self.motion.state == self.motion.STATES['TO_WARD']:
            directions = self.motion.ward_directions.get(4, {})
            if self.motion.target_wards[0] in directions:
                self.motion.turn_direction = directions[self.motion.target_wards[0]]
                self.get_logger().info(f'右4路口，前往病房 {self.motion.target_wards[0]}，转向: {self.motion.turn_direction}')
                self.motion.waiting_at_crossroad = True
                self.motion.enter_crossroad_time = time.time()
                self.motion.turn_history.append(self.motion.turn_direction)
                self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
            else:
                self.get_logger().warn(f'右4路口，病房 {self.motion.target_wards[0]} 的方向信息缺失')
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'十字路口 4 无有效方向信息，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
        elif self.motion.state == self.motion.STATES['TO_SECOND_WARD']:
            if not self.motion.crossroad_4_passed:
                self.get_logger().info(f'左4路口，crossroad_4_passed: {self.motion.crossroad_4_passed}, turn_history: {self.motion.turn_history}')
                if not self.motion.turn_history:
                    self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'左4路口，turn_history 为空，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = -self.motion.turn_history[0]
                    self.get_logger().info(f'左4路口，反转进入病房 {self.motion.target_wards[0]} 的方向，转向: {self.motion.turn_direction}')
                    self.motion.crossroad_4_passed = True
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
            else:
                self.get_logger().info(f'右4路口，crossroad_4_passed: {self.motion.crossroad_4_passed}, turn_history: {self.motion.turn_history}')
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'右4路口，前往第二病房，暂无方向信息，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
        elif self.motion.state == self.motion.STATES['RETURNING']:
            self.get_logger().info(f'返回路径右4路口，turn_history: {self.motion.turn_history}')
            self.motion.turn_direction = -self.motion.turn_history[-1]
            self.get_logger().info(f'返回路径右4路口，反转进入病房 {self.motion.target_wards[1]} 的方向，转向: {self.motion.turn_direction}')
            self.motion.waiting_at_crossroad = True
            self.motion.enter_crossroad_time = time.time()

    def handle_crossroad_5(self):
        if self.motion.state == self.motion.STATES['TO_SECOND_WARD']:
            if self.motion.wards_same_side:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'第五个路口，病房在同一侧，直行前往第二个病房，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
            else:
                if not self.motion.turn_history:
                    self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'第五个路口，turn_history 为空，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = -self.motion.turn_history[1]
                    self.get_logger().info(f'第五个路口，复用第四个路口的方向反转，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))

    def handle_crossroad_6(self):
        if self.motion.state == self.motion.STATES['RETURNING']:
            if self.motion.wards_same_side:
                if len(self.motion.turn_history) < 2:
                    self.get_logger().error('turn_history 不足，无法复用路口 4 方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'第六个路口，turn_history 不足，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = self.motion.turn_history[1]
                    self.get_logger().info(f'第六个路口，复用第四个路口的方向，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
            else:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'十字路口 {self.motion.current_crossroad_id} 跳过病房检测，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')

    def handle_crossroad_7(self):
        if self.motion.state == self.motion.STATES['TO_SECOND_WARD']:
            directions = self.motion.ward_directions.get(7, {})
            if self.motion.target_wards[1] in directions:
                self.motion.turn_direction = directions[self.motion.target_wards[1]]
                self.get_logger().info(f'第七个路口，前往病房 {self.motion.target_wards[1]}，转向: {self.motion.turn_direction}')
            else:
                self.get_logger().warn(f'第七个路口，病房 {self.motion.target_wards[1]} 的方向信息缺失，使用病房 {self.motion.target_wards[0]} 的方向反转')
                self.motion.turn_direction = -directions[self.motion.target_wards[0]]
            self.motion.waiting_at_crossroad = True
            self.motion.enter_crossroad_time = time.time()
            self.motion.turn_history.append(self.motion.turn_direction)
            self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
        elif self.motion.state == self.motion.STATES['RETURNING']:
            if self.motion.wards_same_side:
                if not self.motion.turn_history:
                    self.get_logger().error('turn_history 为空，无法反转路口 3 方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'第七个路口，turn_history 为空，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = -self.motion.turn_history[0]
                    self.get_logger().info(f'第七个路口，反转第三个路口的方向，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))
            else:
                self.motion.turn_direction = -self.motion.turn_history[-1]
                self.get_logger().info(f'返回路径第七个路口，反转进入病房 {self.motion.target_wards[1]} 的方向，转向: {self.motion.turn_direction}')
                self.motion.waiting_at_crossroad = True
                self.motion.enter_crossroad_time = time.time()

    def handle_crossroad_8(self):
        if self.motion.state == self.motion.STATES['RETURNING']:
            if self.motion.wards_same_side:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'第八个路口，病房在同一侧，直行返回药房，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
            else:
                if not self.motion.turn_history:
                    self.get_logger().error('turn_history 为空，无法反转方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'第八个路口，turn_history 为空，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = -self.motion.turn_history[-1]
                    self.get_logger().info(f'第八个路口，复用第七个路口的方向反转，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))

    def handle_crossroad_9(self):
        if self.motion.state == self.motion.STATES['RETURNING']:
            if self.motion.wards_same_side:
                self.motion.forcing_straight = True
                self.motion.straight_start_time = time.time()
                self.get_logger().info(f'第九个路口，病房在同一侧，直行返回药房，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
            else:
                if not self.motion.turn_history:
                    self.get_logger().error('turn_history 为空，无法复用方向，强制直行')
                    self.motion.forcing_straight = True
                    self.motion.straight_start_time = time.time()
                    self.get_logger().info(f'第九个路口，turn_history 为空，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')
                else:
                    self.motion.turn_direction = self.motion.turn_history[0]
                    self.get_logger().info(f'第九个路口，复用第三个路口的方向，转向: {self.motion.turn_direction}')
                    self.motion.waiting_at_crossroad = True
                    self.motion.enter_crossroad_time = time.time()
                    self.motion.turn_history.append(self.motion.turn_direction)
                    self.motion.crossroad_history.append((self.motion.current_crossroad_id, self.motion.turn_direction))

    def handle_crossroad_10_11(self):
        if self.motion.state == self.motion.STATES['RETURNING']:
            self.motion.forcing_straight = True
            self.motion.straight_start_time = time.time()
            self.get_logger().info(f'十字路口 {self.motion.current_crossroad_id} 直行，强制直行 {self.motion.STRAIGHT_WAIT_TIME} 秒')

    def handle_state_logic(self, line_center):
        left_speed = right_speed = 0.0
        if self.motion.forcing_straight:
            elapsed_time = time.time() - self.motion.straight_start_time
            if elapsed_time < self.motion.STRAIGHT_WAIT_TIME:
                self.get_logger().info(f'强制直行中，剩余时间: {self.motion.STRAIGHT_WAIT_TIME - elapsed_time:.2f}秒')
                left_speed, right_speed = self.motion.BASE_SPEED, self.motion.BASE_SPEED
                self.motion.publish_cmd_vel(left_speed, right_speed)
                return left_speed, right_speed
            else:
                self.get_logger().info('强制直行结束，恢复巡线')
                self.motion.forcing_straight = False

        if self.motion.state == self.motion.STATES['WAITING']:
            self.get_logger().info('当前状态为 WAITING，停车')
            self.motion.publish_cmd_vel(0.0, 0.0)
            return 0.0, 0.0

        elif self.motion.state in [self.motion.STATES['TO_WARD'], self.motion.STATES['TO_SECOND_WARD']]:
            if self.motion.waiting_at_crossroad:
                elapsed_time = time.time() - self.motion.enter_crossroad_time
                if elapsed_time < self.motion.WAIT_AT_CROSSROAD_TIME:
                    self.get_logger().info(f'等待到达十字路口中心，剩余时间: {self.motion.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                    left_speed, right_speed = self.motion.BASE_SPEED, self.motion.BASE_SPEED
                else:
                    self.get_logger().info('已到达十字路口中心，开始转向')
                    self.motion.waiting_at_crossroad = False
                    self.motion.turning = True
                    self.motion.entered_crossroad = False
                    self.motion.line_left_center = False
                    self.motion.perform_turn(line_center, self.pid)
            elif self.motion.turning:
                self.motion.perform_turn(line_center, self.pid)
            else:
                if line_center is not None:
                    self.motion.line_lost_count = 0
                    error = line_center - self.motion.image_center
                    pid_output = self.pid.compute(error)
                    left_speed = self.motion.BASE_SPEED + pid_output
                    right_speed = self.motion.BASE_SPEED - pid_output
                    left_speed = max(min(left_speed, self.motion.MAX_SPEED), -self.motion.MAX_SPEED)
                    right_speed = max(min(right_speed, self.motion.MAX_SPEED), -self.motion.MAX_SPEED)
                else:
                    self.motion.line_lost_count += 1
                    if self.motion.line_lost_count > self.motion.MAX_LINE_LOST:
                        self.get_logger().info(f'到达病房 {self.motion.target_wards[self.motion.current_ward_index]}')
                        self.motion.state = self.motion.STATES['UNLOADING']
                        self.motion.unload_start_time = time.time()
                        self.motion.line_lost_count = 0
                        self.motion.publish_cmd_vel(0.0, 0.0)
                        return 0.0, 0.0
                    left_speed = right_speed = self.motion.BASE_SPEED
                self.motion.publish_cmd_vel(left_speed, right_speed)

        elif self.motion.state == self.motion.STATES['UNLOADING']:
            if time.time() - self.motion.unload_start_time >= self.motion.UNLOAD_TIME:
                self.get_logger().info('卸载药品完成')
                self.motion.state = self.motion.STATES['TURNING_AROUND']
                self.motion.turn_around_start_time = time.time()
                self.get_logger().info('开始180°掉头')
            self.motion.publish_cmd_vel(0.0, 0.0)

        elif self.motion.state == self.motion.STATES['TURNING_AROUND']:
            if time.time() - self.motion.turn_around_start_time >= self.motion.TURN_AROUND_TIME:
                self.get_logger().info('掉头完成')
                self.motion.current_ward_index += 1
                if self.motion.current_ward_index < len(self.motion.target_wards):
                    self.motion.state = self.motion.STATES['TO_SECOND_WARD']
                    self.motion.last_crossroad_id = 0
                    self.motion.crossroad_4_passed = False
                    self.get_logger().info(f'前往下一个病房: {self.motion.target_wards[self.motion.current_ward_index]}')
                    self.get_logger().info(f'进入 TO_SECOND_WARD 状态，crossroad_4_passed: {self.motion.crossroad_4_passed}')
                else:
                    self.motion.state = self.motion.STATES['RETURNING']
                    self.motion.return_turn_index = len(self.motion.turn_history) - 1
                    self.motion.last_crossroad_id = 0
                    self.get_logger().info(f'开始返回药房，turn_history: {self.motion.turn_history}')
            else:
                self.motion.perform_turn_around()

        elif self.motion.state == self.motion.STATES['RETURNING']:
            if self.motion.waiting_at_crossroad:
                elapsed_time = time.time() - self.motion.enter_crossroad_time
                if elapsed_time < self.motion.WAIT_AT_CROSSROAD_TIME:
                    self.get_logger().info(f'等待到达十字路口中心，剩余时间: {self.motion.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                    left_speed, right_speed = self.motion.BASE_SPEED, self.motion.BASE_SPEED
                else:
                    self.get_logger().info('已到达十字路口中心，开始转向')
                    self.motion.waiting_at_crossroad = False
                    self.motion.turning = True
                    self.motion.entered_crossroad = False
                    self.motion.line_left_center = False
                    self.motion.perform_turn(line_center, self.pid)
            elif self.motion.turning:
                self.motion.perform_turn(line_center, self.pid)
            else:
                if line_center is not None:
                    self.motion.line_lost_count = 0
                    error = line_center - self.motion.image_center
                    pid_output = self.pid.compute(error)
                    left_speed = self.motion.BASE_SPEED + pid_output
                    right_speed = self.motion.BASE_SPEED - pid_output
                    left_speed = max(min(left_speed, self.motion.MAX_SPEED), -self.motion.MAX_SPEED)
                    right_speed = max(min(right_speed, self.motion.MAX_SPEED), -self.motion.MAX_SPEED)
                else:
                    self.motion.line_lost_count += 1
                    if self.motion.line_lost_count > self.motion.MAX_LINE_LOST:
                        self.get_logger().info(f'返回药房，总用时: {time.time() - self.motion.start_time:.2f}s')
                        self.motion.state = self.motion.STATES['WAITING']
                        self.motion.current_ward_index = 0
                        self.motion.turn_history = []
                        self.motion.crossroad_history = []
                        self.motion.return_turn_index = 0
                        self.motion.current_crossroad_id = 0
                        self.motion.last_crossroad_id = 0
                        self.get_logger().info('到达药房，停车')
                        self.motion.publish_cmd_vel(0.0, 0.0)
                        self.motion.mode = 1
                        return 0.0, 0.0
                    left_speed = right_speed = self.motion.BASE_SPEED
                self.motion.publish_cmd_vel(left_speed, right_speed)

        return left_speed, right_speed
