import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        self.bridge = CvBridge()
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        self.declare_parameter('video_device', '/dev/video0')  # 摄像头设备
        self.declare_parameter('frame_width', 640)             # 图像宽度
        self.declare_parameter('frame_height', 480)            # 图像高度
        self.declare_parameter('frame_rate', 30.0)             # 帧率
        
        # 获取参数
        self.video_device = self.get_parameter('video_device').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.frame_rate = self.get_parameter('frame_rate').value
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头: {self.video_device}')
            raise RuntimeError('摄像头初始化失败')
        
        # 设置摄像头分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        # 验证分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'摄像头初始化成功: 分辨率 {actual_width}x{actual_height}, 帧率 {actual_fps} fps'
        )
        
        # 创建定时器，定期发布图像
        timer_period = 1.0 / self.frame_rate  # 计算帧间隔（秒）
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('摄像头发布节点已启动，发布到 /camera/image_raw')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('无法读取摄像头帧')
            return
        
        try:
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = 'camera_frame'

            self.image_pub.publish(image_msg)
            
            # 可视化（调试用）
            #cv2.imshow('Camera Feed', frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'图像转换或发布失败: {str(e)}')

    def destroy_node(self):
        # 释放摄像头资源
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点已停止')
    except RuntimeError as e:
        node.get_logger().error(f'启动失败: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()