import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray

class WardPublisherNode(Node):
    def __init__(self):
        super().__init__('targer')
        self.publisher_ = self.create_publisher(Int8MultiArray, '/target_wards', 10)
        self.target_wards = [4,8]  # 静态配置，可改为动态输入
        self.timer = self.create_timer(1.0, self.publish_wards)  # 每秒发布
        self.get_logger().info('病房发布节点已启动')

    def publish_wards(self):
        msg = Int8MultiArray()
        msg.data = [int(ward) for ward in self.target_wards]
        self.publisher_.publish(msg)
        #self.get_logger().info(f'发布目标病房: {self.target_wards}')

def main(args=None):
    rclpy.init(args=args)
    node = WardPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点已停止')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()