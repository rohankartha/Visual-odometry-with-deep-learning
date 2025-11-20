#!/usr/bin/env python

# import of relevant libraries.
import rclpy # module for ROS APIs
from rclpy.node import Node


from rclpy.qos import QoSProfile

from message_filters import Subscriber, ApproximateTimeSynchronizer

from rclpy.time import Time

# Importing ROS message types
from sensor_msgs.msg import Image, CameraInfo

from rclpy.serialization import deserialize_message

import rosbag2_py
import yaml
import time

from std_msgs.msg import String
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Time
from collections import defaultdict

from pprint import pprint

# Constants.
FREQUENCY = 20 #Hz.
LINEAR_VELOCITY = 0.125 #m/s
DURATION = 2.0 #s how long the message should be published.
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'scan'
DEFAULT_ODOM_TOPIC = 'odom'
USE_SIM_TIME = True
IMAGE_TOPIC_LEFT = "/cam0/image_raw"
CAMERA_TOPIC_LEFT = "/cam0/camera_info"
IMAGE_TOPIC_RIGHT = "/cam1/image_raw"
CAMERA_TOPIC_RIGHT = "/cam1/camera_info"
START_PUBLISH_TOPIC = "publish_start"


class ImagePublisher(Node):
    def __init__(self, image_dir: str, config_path: str):

        super().__init__("image_publisher")

        # Initializing profile to govern message quality, history, etc.
        qos = QoSProfile(depth=10)

        # Node subscribes to publish start topic
        self.create_subscription(msg_type=String, topic=START_PUBLISH_TOPIC, callback=self._start_callback, qos_profile=qos)


        # Load dataset directory and configuration
        self._image_dir = image_dir
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)
        
        # Construct camera info message
        config = config["cam0"]
        camera_info_msg = CameraInfo()
        camera_info_msg.width = config["resolution"][0]
        camera_info_msg.height = config["resolution"][1]
        camera_info_msg.distortion_model = config["distortion_model"]
        camera_info_msg.d = config["distortion_coeffs"]

        fu, fv, cu, cv = config["intrinsics"]
        camera_info_msg.k = [fu, 0.0, cu, 
                             0.0, fv, cv, 
                             0.0, 0.0, 1.0]
        camera_info_msg.r = [1.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 
                             0.0, 0.0, 1.0]
        camera_info_msg.p = [fu, 0.0, cu, 0.0, 
                             0.0, fv, cv, 0.0, 
                             0.0, 0.0, 1.0, 0.0]
        self.camera_info_msg_template = camera_info_msg
        
        qos = QoSProfile(depth=10)
        
        # Node publishes to image_raw and camera_info
        self._left_cam_image_pub = self.create_publisher(msg_type=Image, topic=IMAGE_TOPIC_LEFT, qos_profile=qos)
        self._right_cam_image_pub = self.create_publisher(msg_type=Image, topic=IMAGE_TOPIC_RIGHT, qos_profile=qos)
        self._left_cam_info_pub = self.create_publisher(msg_type=CameraInfo, topic=CAMERA_TOPIC_LEFT, qos_profile=qos)
        self._right_cam_info_pub = self.create_publisher(msg_type=CameraInfo, topic=CAMERA_TOPIC_RIGHT, qos_profile=qos)

        return


    def _start_callback(self, msg):
        self.get_logger().info("Start message received")
        left_cam_image_messages, right_cam_image_messages = self.read_bag()
        self.publish_messages(left_cam_image_messages, right_cam_image_messages)
    

    def read_bag(self):

        # Open bag reader
        storage_options = rosbag2_py.StorageOptions(uri=self._image_dir, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions()
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Create data structures to hold parsed messages
        left_cam_image_messages = []
        right_cam_image_messages = []

        # Iterate through all messages
        while reader.has_next():
            topic, msg, _ = reader.read_next()

            # Case 1: Camera message
            if "image_raw" in topic:
                if topic == IMAGE_TOPIC_LEFT:
                    camera_type = "left"
                else:
                    camera_type = "right"

                # Parse image message
                image_msg = deserialize_message(serialized_message=msg, message_type=Image)

                # Append messages to corresponding array
                if camera_type == "left":
                    left_cam_image_messages.append(image_msg)
                else:
                    right_cam_image_messages.append(image_msg)
                    
            # # Case 2: imu message
            # elif "imu" in topic:

            # may have to interpolate inertial readings

        self.get_logger().info("Finished loading dataset")
        
        return left_cam_image_messages, right_cam_image_messages


    def publish_messages(self, left_cam_image_messages: list, right_cam_image_messages: list):

        num_images = len(left_cam_image_messages)

        for i in range(num_images):
            left_cam_image = left_cam_image_messages[i]
            right_cam_image = right_cam_image_messages[i]

            # get cam info template
            left_cam_info = self.camera_info_msg_template
            left_cam_info.header.frame_id = "cam0"
            left_cam_info.header.stamp = left_cam_image.header.stamp
            right_cam_info = self.camera_info_msg_template
            right_cam_info.header.frame_id = "cam1"
            right_cam_info.header.stamp = right_cam_image.header.stamp


            self._left_cam_image_pub.publish(left_cam_image)
            self._right_cam_image_pub.publish(right_cam_image)
            self._left_cam_info_pub.publish(left_cam_info)
            self._right_cam_info_pub.publish(right_cam_info)

            time.sleep(0.05)
        
        return
    
    