import time

class DualWardMode:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger()

    def handle(self, line_center, contours_upper, contours_left, contours_right):
        left_speed = right_speed = 0.0

        if not self.node.turning and not self.node.waiting_at_crossroad and self.node.detect_crossroad(contours_upper, contours_left, contours_right, self.node.IMAGE_WIDTH):
            self.node.crossroad_count += 1
            if self.node.crossroad_count == 1:
                self.node.current_crossroad_id = 2
            else:
                self.node.current_crossroad_id += 1

            self.logger.info(f'TO_SECOND_WARD 检测到十字路口 {self.node.current_crossroad_id} (计数: {self.node.crossroad_count})')

            crossroad_already_passed = False
            for crossroad_id, _ in self.node.crossroad_history:
                if crossroad_id == self.node.current_crossroad_id:
                    crossroad_already_passed = True
                    break

            if crossroad_already_passed:
                for crossroad_id, turn in self.node.crossroad_history:
                    if crossroad_id == self.node.current_crossroad_id:
                        self.node.lock_turn_direction = True
                        self.node.turn_direction = turn
                        break
                self.logger.info(f'十字路口 {self.node.current_crossroad_id} 已重复经过，重用转向: {self.node.turn_direction}')
            else:
                self.logger.info(f'十字路口 {self.node.current_crossroad_id}，使用新转向: {self.node.turn_direction}')

            if self.node.turn_direction != 0:
                self.logger.info(f'十字路口 {self.node.current_crossroad_id}，执行转向: {self.node.turn_direction}')
                self.node.waiting_at_crossroad = True
                self.node.enter_crossroad_time = time.time()
                if not crossroad_already_passed:
                    self.node.turn_history.append(self.node.turn_direction)
                    self.node.crossroad_history.append((self.node.current_crossroad_id, self.node.turn_direction))
                    self.logger.info(f'新增转向记录: {self.node.turn_direction}，turn_history: {self.node.turn_history}，crossroad_history: {self.node.crossroad_history}')
                left_speed, right_speed = self.node.BASE_SPEED, self.node.BASE_SPEED
            else:
                self.logger.info(f'十字路口 {self.node.current_crossroad_id}，直行')
                if not crossroad_already_passed:
                    self.node.turn_history.append(0)
                    self.node.crossroad_history.append((self.node.current_crossroad_id, 0))
                    self.logger.info(f'新增直行记录: 0，turn_history: {self.node.turn_history}，crossroad_history: {self.node.crossroad_history}')
                self.node.publish_cmd_vel(self.node.BASE_SPEED, self.node.BASE_SPEED)
                time.sleep(self.node.STRAIGHT_WAIT_TIME)
                return left_speed, right_speed

        elif self.node.waiting_at_crossroad:
            elapsed_time = time.time() - self.node.enter_crossroad_time
            if elapsed_time < self.node.WAIT_AT_CROSSROAD_TIME:
                self.logger.info(f'等待到达十字路口中心，剩余时间: {self.node.WAIT_AT_CROSSROAD_TIME - elapsed_time:.2f}秒')
                left_speed, right_speed = self.node.BASE_SPEED, self.node.BASE_SPEED
            else:
                self.logger.info('已到达十字路口中心，开始转向')
                self.node.waiting_at_crossroad = False
                self.node.turning = True
                self.node.entered_crossroad = False
                self.node.line_left_center = False
                self.node.perform_turn(line_center)

        elif self.node.turning:
            self.node.perform_turn(line_center)
            if not self.node.turning:
                self.node.lock_turn_direction = False
                self.logger.info('转向完成，解锁 turn_direction')

        else:
            if line_center is not None:
                self.node.line_lost_count = 0
                error = line_center - self.node.image_center
                pid_output = self.node.pid.compute(error)
                left_speed = self.node.BASE_SPEED + pid_output
                right_speed = self.node.BASE_SPEED - pid_output
                left_speed = max(min(left_speed, self.node.MAX_SPEED), -self.node.MAX_SPEED)
                right_speed = max(min(right_speed, self.node.MAX_SPEED), -self.node.MAX_SPEED)
            else:
                self.node.line_lost_count += 1
                if self.node.line_lost_count > self.node.MAX_LINE_LOST:
                    self.logger.info(f'到达病房 {self.node.target_wards[self.node.current_ward_index]}')
                    self.node.state = self.node.STATES['UNLOADING']
                    self.node.unload_start_time = time.time()
                    self.node.line_lost_count = 0
                    self.node.publish_cmd_vel(0.0, 0.0)
                    return 0.0, 0.0
                left_speed = right_speed = self.node.BASE_SPEED
            self.node.publish_cmd_vel(left_speed, right_speed)

        return left_speed, right_speed