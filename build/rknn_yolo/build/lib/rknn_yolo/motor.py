import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float32MultiArray, Float32, Int8
import time
import re

class SerialNode(Node):
    def __init__(self):
        super().__init__('motor')
        
        self.serial_port = serial.Serial(
            port='/dev/ttyS1',
            baudrate=9600,
            timeout=1
        )
        
        self.led_sub = self.create_subscription(
            Int8,
            '/led',
            self.led_callback,
            10
        )
        self.publisher_ = self.create_publisher(Float32, '/serial_mode', 10)
        self.cmd_vel_sub = self.create_subscription(
            Float32MultiArray,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.mode = 1
        self.flag = True
        self.timer = self.create_timer(0.1, self.read_serial)  # 每0.1秒读取串口
        #self.get_logger().info('Serial node started, listening for /cmd_vel messages...')

    def led_callback(self, msg):
        led_data = msg.data
        if led_data != 4:
            frame = bytearray([
                    0xC7, 0xC8,
                    0x00,led_data,
                    0x5D
                ])
            self.get_logger().info(f"led: '{led_data}'")
            
            self.serial_port.write(frame)
        else :
            frame = bytearray([
                    0xC7, 0xC8,
                    0x00,0x00,
                    0x5D
                ])
            #self.get_logger().info("led: '0'")
            self.serial_port.write(frame)


    def read_serial(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.get_logger().warn("Serial port not available")
            return

        try:
            # 读取串口数据
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.readline().decode('utf-8', errors='ignore').rstrip()
                self.get_logger().debug(f"Raw serial data: '{data}'")

                # 解析 "HX711 Get Weight = <float>" 格式
                match = re.match(r'HX711 Get Weight = (\d+\.\d+)', data)
                if match:
                    float_value = float(match.group(1))
                    self.publisher_.publish(Float32(data=float_value))
                    self.get_logger().info(f"Published weight to /serial_mode: {float_value}")
        except Exception as e:
            self.get_logger().error(f"Error reading serial: {e}")

    def cmd_vel_callback(self, msg):
        if len(msg.data) != 3:
            self.get_logger().error(f"收到无效的轮速数据，长度应为 3，实际为 {len(msg.data)}")
            return

        new_mode = int(msg.data[2])
        #self.get_logger().info(f"new_modee:{new_mode}")
        if new_mode != self.mode:
            if new_mode == 2:
                self.flag = True  # 重置flag以允许发送启动帧
                self.get_logger().info("Mode switched to 2, reset flag to True")
            self.mode = new_mode
            self.get_logger().info(f"Mode updated to {self.mode}")

        if self.mode == 2:
            if self.flag:
                frame = bytearray([0xC5, 0xC6, 0x00, 0x00, 0x4D])
                self.serial_port.write(frame)
                #self.get_logger().info("Sent start frame: [0xC5, 0xC6, 0x00, 0x00, 0x4D]")
                self.flag = False

            left_speed = msg.data[0]  # 左轮速度（cm/s）
            right_speed = msg.data[1]  # 右轮速度（cm/s）

            motor1_dir = 0 if left_speed >= 0 else 1  # 电机1方向（左轮）
            motor2_dir = 1 if right_speed >= 0 else 0  # 电机2方向（右轮）

            motor1_speed = abs(int(left_speed))  # 电机1速度
            motor2_speed = abs(int(right_speed))  # 电机2速度

            motor1_speed_high = (motor1_speed >> 8) & 0xFF
            motor1_speed_low = motor1_speed & 0xFF
            motor2_speed_high = (motor2_speed >> 8) & 0xFF
            motor2_speed_low = motor2_speed & 0xFF

            frame = bytearray([
                0xA5, 0xA6,
                motor1_dir,
                motor2_dir,
                motor1_speed_high,
                motor1_speed_low,
                motor2_speed_high,
                motor2_speed_low,
                0x6B, 0x5B
            ])
            self.serial_port.write(frame)
            #self.get_logger().info(f"Sent motor frame: left_speed={left_speed}, right_speed={right_speed}")
            time.sleep(0.02)

        elif self.mode == 1:
            frame = bytearray([0xC5, 0xC6, 0x00, 0x01, 0x4D])
            self.serial_port.write(frame)
            #self.get_logger().info("Sent quality check frame: [0xC5, 0xC6, 0x00, 0x01, 0x4D]")
            time.sleep(0.02)

    def __del__(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info('Serial port closed.')

def main(args=None):
    rclpy.init(args=args)
    node = SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()