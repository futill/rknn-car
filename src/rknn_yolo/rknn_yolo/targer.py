import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray
import serial

class WardPublisherNode(Node):
    def __init__(self):
        super().__init__('targer')  # 节点名称保持为 'targer'
        # 初始化串口
        self.serial_port = serial.Serial(
            port='/dev/ttyS4',
            baudrate=115200,
            timeout=1
        )
        self.publisher_ = self.create_publisher(Int8MultiArray, '/target_wards', 10)
        self.target_wards = []  # 动态存储病房号
        self.timer = self.create_timer(0.1, self.read_and_publish_wards)  # 每 0.1 秒检查串口数据
        self.get_logger().info('病房发布节点已启动')

    def read_and_publish_wards(self):
            # 读取串口数据
            if self.serial_port.in_waiting == 8:  # 协议长度为 4 字节
                data = self.serial_port.read(8)  # 读取 4 字节
                # 验证协议格式：aa 01 (病房号) bb
                if len(data) == 8 and data[0] == 0xAA and data[1] == 0x02 and data[7] == 0xBB:
                    ward_number_one = data[2]  # 提取病房号（第 3 字节）
                    ward_number_double = data[6]
                    self.target_wards = [ward_number_one,ward_number_double]  # 更新病房号列表
                    # 发布消息
                    msg = Int8MultiArray()
                    msg.data = [int(ward) for ward in self.target_wards]
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'发布病房号: {self.target_wards}')
                    data = []
                else:
                    self.get_logger().warn('收到无效的串口数据')
            elif self.serial_port.in_waiting == 4:  # 协议长度为 4 字节
                data = self.serial_port.read(4)  # 读取 4 字节
                # 验证协议格式：aa 01 (病房号) bb
                if len(data) == 4 and data[0] == 0xAA and data[1] == 0x01 and data[3] == 0xBB:
                    ward_number_one = data[2]  # 提取病房号（第 3 字节）
                    self.target_wards = [ward_number_one]  # 更新病房号列表
                    # 发布消息
                    msg = Int8MultiArray()
                    msg.data = [int(ward) for ward in self.target_wards]
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'发布病房号: {self.target_wards}')
                    data = []
                else:
                    self.get_logger().warn('收到无效的串口数据')


def main(args=None):
    rclpy.init(args=args)
    node = WardPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点已停止')
    except Exception as e:
        node.get_logger().error(f'发生错误: {e}')
    finally:
        if node.serial_port.is_open:
            node.serial_port.close()  # 关闭串口
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()