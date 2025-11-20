# Dependencies
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from src.conventional.odometry.stereo_vo import StereoVO
from src.conventional.odometry.stereo_vio import StereoVIO
from image_publisher import ImagePublisher
from rclpy.qos import QoSProfile

from std_msgs.msg import String
import numpy as np

# Topics
START_PUBLISH_TOPIC = "publish_start"
SHUTDOWN_TOPIC = "shutdown"
USER_START_TOPIC = "user_start"

class Controller(Node):

    def __init__(self, executor):

        super().__init__("Controller")
        self.executor = executor

        # Initializing profile to govern message quality, history, etc.
        qos = QoSProfile(depth=10)

        # Node publishes to publish start topic
        self.start_publish = self.create_publisher(msg_type=String, topic=START_PUBLISH_TOPIC, qos_profile=qos)

        # Node subscribes to shutdown topic
        self.create_subscription(msg_type=String, topic=USER_START_TOPIC, callback=self._start_callback, qos_profile=qos)
        self.create_subscription(msg_type=String, topic=SHUTDOWN_TOPIC, callback=self._shutdown_callback, qos_profile=qos)
    
    def _start_callback(self, msg):

        # Publish message to custom topic to start image publishing
        msg = String()
        msg.data = "Start publishing"
        self.start_publish.publish(msg=msg)
        self.get_logger().info("Start message published")


    # callback to shutdown the nodes
    def _shutdown_callback(self, msg: String):
        if msg.data == "shutdown":
            self.executor.shutdown()

if __name__ == "__main__":

    rclpy.init()

    executor = MultiThreadedExecutor()

    # Experiment node
    controller = Controller(executor=executor)

    # From dataset ground truth
    initial_position = np.array([0.878895, 2.183400, 0.948427])
    initial_quaternion = np.array([-0.824237, -0.106942, -0.551702, 0.069433])

    # Odometry node
    stereo_vo_node = StereoVO(config_path="data/camera_config.yaml",
                              initial_position=initial_position,
                              initial_quaternion=initial_quaternion)
    
    # Stereo Visual-Inertial Odometry
    # stereo_vio_node = StereoVIO(camera_config="data/camera_config.yaml",
    #                             imu_config="data/kalibr_imu_chain.yaml",
    #                             initial_position=initial_position,
    #                             initial_quaternion=initial_quaternion)

    # Image publisher node
    publisher = ImagePublisher(image_dir="data/V1_01_easy.db3",
                               config_path="data/camera_config.yaml")
    
    executor.add_node(controller)
    executor.add_node(stereo_vo_node)
    executor.add_node(publisher)

    try:
        executor.spin()
    except KeyboardInterrupt as e:
        print("Keyboard interrupt detected")
    finally:
        executor.shutdown()
        rclpy.shutdown()