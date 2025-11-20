#!/usr/bin/env python

# import of relevant libraries.
import rclpy # module for ROS APIs
from rclpy.node import Node


from rclpy.qos import QoSProfile

from message_filters import Subscriber, ApproximateTimeSynchronizer

from rclpy.time import Time

# Importing ROS message types
from sensor_msgs.msg import Image, CameraInfo, Imu

from rclpy.serialization import deserialize_message

import rosbag2_py
import yaml
import time

from std_msgs.msg import String
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Time
from collections import defaultdict

from pprint import pprint
import copy

# Constants.
FREQUENCY = 20 #Hz.
DURATION = 2.0 #s how long the message should be published.
USE_SIM_TIME = True
IMAGE_TOPIC_LEFT = "/cam0/image_raw"
CAMERA_TOPIC_LEFT = "/cam0/camera_info"
IMAGE_TOPIC_RIGHT = "/cam1/image_raw"
CAMERA_TOPIC_RIGHT = "/cam1/camera_info"
START_PUBLISH_TOPIC = "publish_start"
IMU_TOPIC = "/imu/data"


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
            cam_config = yaml.safe_load(fp)
        
        # Construct camera info message for left camera
        config = cam_config["cam0"]
        camera_info_msg_0 = CameraInfo()
        camera_info_msg_0.width = config["resolution"][0]
        camera_info_msg_0.height = config["resolution"][1]
        camera_info_msg_0.distortion_model = config["distortion_model"]
        camera_info_msg_0.d = config["distortion_coeffs"]

        fu, fv, cu, cv = config["intrinsics"]
        camera_info_msg_0.k = [fu, 0.0, cu, 
                             0.0, fv, cv, 
                             0.0, 0.0, 1.0]
        camera_info_msg_0.r = [1.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 
                             0.0, 0.0, 1.0]
        camera_info_msg_0.p = [fu, 0.0, cu, 0.0, 
                             0.0, fv, cv, 0.0, 
                             0.0, 0.0, 1.0, 0.0]
        self.camera_info_msg_0_template = camera_info_msg_0

        # Construct camera info message for right camera
        config = cam_config["cam1"]
        camera_info_msg_1 = CameraInfo()
        camera_info_msg_1.width = config["resolution"][0]
        camera_info_msg_1.height = config["resolution"][1]
        camera_info_msg_1.distortion_model = config["distortion_model"]
        camera_info_msg_1.d = config["distortion_coeffs"]

        fu, fv, cu, cv = config["intrinsics"]
        camera_info_msg_1.k = [fu, 0.0, cu, 
                             0.0, fv, cv, 
                             0.0, 0.0, 1.0]
        camera_info_msg_1.r = [1.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 
                             0.0, 0.0, 1.0]
        camera_info_msg_1.p = [fu, 0.0, cu, 0.0, 
                             0.0, fv, cv, 0.0, 
                             0.0, 0.0, 1.0, 0.0]
        self.camera_info_msg_1_template = camera_info_msg_1
        
        qos = QoSProfile(depth=10)
        
        # Node publishes to image_raw and camera_info
        self._left_cam_image_pub = self.create_publisher(msg_type=Image, topic=IMAGE_TOPIC_LEFT, qos_profile=qos)
        self._right_cam_image_pub = self.create_publisher(msg_type=Image, topic=IMAGE_TOPIC_RIGHT, qos_profile=qos)
        self._left_cam_info_pub = self.create_publisher(msg_type=CameraInfo, topic=CAMERA_TOPIC_LEFT, qos_profile=qos)
        self._right_cam_info_pub = self.create_publisher(msg_type=CameraInfo, topic=CAMERA_TOPIC_RIGHT, qos_profile=qos)

        # Node publishes to imu topic
        self._imu_publisher = self.create_publisher(msg_type=Imu, topic=IMU_TOPIC, qos_profile=qos)

        # 
        self.image_counter = 0
        self.imu_counter = 0

        # Initializing variables to hold messages from bag
        self.left_cam_image_messages = None
        self.right_cam_image_messages = None
        self.imu_messages = None

        return


    def _start_callback(self, msg):
        self.get_logger().info("Start message received")

        # Load messages from bag
        self.read_bag()

        # register
        self.image_timer = self.create_timer(0.05, self.publish_image_message)   
        self.imu_timer = self.create_timer(0.005, self.publish_imu_message)   
    

    def read_bag(self):

        # Open bag reader
        storage_options = rosbag2_py.StorageOptions(uri=self._image_dir, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions()
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Create data structures to hold parsed messages
        left_cam_image_messages = []
        right_cam_image_messages = []
        imu_messages = []

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
                    
            # Case 2: imu message
            elif "imu" in topic:

                # Parse imu message
                imu_msg = deserialize_message(serialized_message=msg, message_type=Imu)
                imu_messages.append(imu_msg)

        self.get_logger().info(f"{len(left_cam_image_messages)} left cam images loaded")
        self.get_logger().info(f"{len(right_cam_image_messages)} right cam images loaded")

        self.left_cam_image_messages = left_cam_image_messages
        self.right_cam_image_messages = right_cam_image_messages
        self.imu_messages = imu_messages

        return


    def publish_image_message(self):

        if self.image_counter >= len(self.left_cam_image_messages):
            self.get_logger().info("Finished publishing all image messages")

            if self.image_timer is not None:
                self.image_timer.cancel()
            return

        # Retrieve image messages
        left_cam_image = self.left_cam_image_messages[self.image_counter]
        right_cam_image = self.right_cam_image_messages[self.image_counter]

        # Update image headers
        left_cam_image.header.frame_id = "cam0"
        right_cam_image.header.frame_id = "cam1"

        # get cam info template
        left_cam_info = copy.deepcopy(self.camera_info_msg_0_template)
        left_cam_info.header.frame_id = "cam0"
        left_cam_info.header.stamp = left_cam_image.header.stamp
        right_cam_info = copy.deepcopy(self.camera_info_msg_1_template)
        right_cam_info.header.frame_id = "cam1"
        right_cam_info.header.stamp = right_cam_image.header.stamp

        # Publish camera messages
        self._left_cam_image_pub.publish(left_cam_image)
        self._right_cam_image_pub.publish(right_cam_image)
        self._left_cam_info_pub.publish(left_cam_info)
        self._right_cam_info_pub.publish(right_cam_info)

        # Increment image counter
        self.image_counter = self.image_counter + 1
        
        return
    

    def publish_imu_message(self):

        if self.imu_counter >= len(self.imu_messages):
            self.get_logger().info("Finished publishing all imu messages")

            if self.imu_timer is not None:
                self.imu_timer.cancel()
            return

        # Retrieve imu message
        imu_message = self.imu_messages[self.imu_counter]

        # Publish message
        self._imu_publisher.publish(msg=imu_message)

        # Increment counter
        self.imu_counter = self.imu_counter + 1

        return



    
    
    