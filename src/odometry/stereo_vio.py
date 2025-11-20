#!/usr/bin/env python

# Importing core ROS dependencies
import rclpy # module for ROS APIs
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# Importing ROS message dependencies
from sensor_msgs.msg import Image, CameraInfo, Imu
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Importing external dependencies
import cv2 as cv
from cv_bridge import CvBridge
import gtsam
import numpy as np
import yaml
import csv
from collections import deque

# Constants
NODE_NAME = "stereo_vio"
USE_SIM_TIME = True
QUEUE_SIZE = 10
MAX_DELAY = 0.05

# Topics
IMAGE_TOPIC_LEFT = "/cam0/image_raw"
CAMERA_TOPIC_LEFT = "/cam0/camera_info"
IMAGE_TOPIC_RIGHT = "/cam1/image_raw"
CAMERA_TOPIC_RIGHT = "/cam1/camera_info"
IMU_TOPIC = "/imu/data"


class StereoVIO(Node):

    ###################
    ### Constructor ###
    ###################

    def __init__(self, camera_config, imu_config, initial_position, initial_quaternion,
                 node_name=NODE_NAME, context=None):
        
        # FOR DEBUGGING
        np.set_printoptions(suppress=True, precision=6)
        
        # Initialize Node super class
        super().__init__(node_name, context=context)

        # Retrieve camera-camera transformation matrix
        with open(camera_config, "r") as fp:
            config = yaml.safe_load(fp)
        cam0_T_imu = np.array(config["cam0"]["T_imu_cam"]).reshape(4,4)
        cam1_T_imu = np.array(config["cam1"]["T_imu_cam"]).reshape(4,4)
        cam0_T_cam1 = np.linalg.inv(cam0_T_imu) @ cam1_T_imu
        cam0_Rtn_cam1 = cam0_T_cam1[:3,:3]
        cam0_Tln_cam1 = cam0_T_cam1[:3,3].reshape(3,1)

        self.R = cam0_Rtn_cam1
        self.T = cam0_Tln_cam1

        # Initialize qos profile
        qos = QoSProfile(depth=10)

        # Create callback group
        self.mutually_excl_cb_group = MutuallyExclusiveCallbackGroup()

        # Synchronous subscriptions to image + camera topics
        self._left_image_sync_sub = Subscriber(self, 
                                          Image, 
                                          IMAGE_TOPIC_LEFT, 
                                          qos_profile=qos, 
                                          callback_group=self.mutually_excl_cb_group)
        self._right_image_sync_sub = Subscriber(self, 
                                          Image, 
                                          IMAGE_TOPIC_RIGHT, 
                                          qos_profile=qos, 
                                          callback_group=self.mutually_excl_cb_group)
        self._left_camera_sync_sub = Subscriber(self,
                                           CameraInfo,
                                           CAMERA_TOPIC_LEFT,
                                           qos_profile=qos,
                                           callback_group=self.mutually_excl_cb_group)
        self._right_camera_sync_sub = Subscriber(self,
                                           CameraInfo,
                                           CAMERA_TOPIC_RIGHT,
                                           qos_profile=qos,
                                           callback_group=self.mutually_excl_cb_group)

        # Synchronize image, depth, and camera messages
        queue_size = QUEUE_SIZE
        max_delay = MAX_DELAY
        self.time_sync = ApproximateTimeSynchronizer(
            [self._left_image_sync_sub,
             self._right_image_sync_sub,
             self._left_camera_sync_sub,
             self._right_camera_sync_sub],
            queue_size=queue_size,
            slop=max_delay
        )

        # Subscribe to imu topic
        self.create_subscription(msg_type=Imu, topic=IMU_TOPIC, callback=self.imu_callback, qos_profile=qos)




        # Create member variables to hold features from previous frames
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Create member variables to hold global rotation + translation matrices
        self.R_global = None
        self.t_global = None

        # Create graph for pose optimization
        self.isam = gtsam.ISAM2()

        self.bridge = CvBridge()
        self.detector = cv.SIFT_create(nfeatures=2000)

        # Register callback for synchronized messages
        self.time_sync.registerCallback(self.synced_camera_callback)
        self.get_logger().info("Time synchronized callbacks registered")

        self.first_frame = True

        self.initial_position = initial_position
        self.initial_quaternion = initial_quaternion
        self.frame_number = -1

        # Maps for result collection
        self.symbol_to_time = dict()

        # Retrieve imu parameters
        with open(imu_config, "r") as fp:
            config = yaml.safe_load(fp)
        accel_noise_density = config["imu0"]["accelerometer_noise_density"]
        accel_rw_bias = config["imu0"]["accelerometer_random_walk"]
        gyro_noise_density = config["imu0"]["gyroscope_noise_density"]
        gyro_rw_bias = config["imu0"]["gyroscope_random_walk"]

        accel_covar = np.eye(3) * accel_noise_density**2
        gyro_covar = np.eye(3) * gyro_noise_density**2
        acc_covar = np.eye(3) * accel_rw_bias**2
        omega_covar = np.eye(3) * gyro_rw_bias**2

        # Create imu params object
        imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        imu_params.setAccelerometerCovariance(accel_covar)
        imu_params.setGyroscopeCovariance(gyro_covar)
        imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
        self.imu_params = imu_params

        # Initialize bias
        bias = gtsam.imuBias.ConstantBias()
        self.bias = bias

        self.prev_vel = np.zeros(3)
        self.imu_buffer = deque()
        self.curr_time = None
        self.prev_time = None

        self.prev_position = initial_position
        self.prev_orientation = initial_quaternion
    
    ###################
    #### Callbacks ####
    ###################
    
    def synced_camera_callback(self, 
                               left_image_msg: Image, 
                               right_image_msg: Image, 
                               left_camera_msg: CameraInfo, 
                               right_camera_msg: CameraInfo):
        
        self.frame_number = self.frame_number + 1
        a = self.frame_number
        self.get_logger().info(f"Processing frame pair {self.frame_number}")

        curr_time = left_image_msg.header.stamp.sec + (left_image_msg.header.stamp.nanosec * 1e-9)

        if a == 50:

            self._print_predictions()
        
        # Extract left camera intrinsic matrix
        k_l = np.array(left_camera_msg.k).reshape(3,3)
        k_r = np.array(right_camera_msg.k).reshape(3,3)
        d_l = np.array(left_camera_msg.d)
        d_r = np.array(right_camera_msg.d)

        # Step 1: Preprocess image
        rect_left, rect_right = self._image_preprocessing(left_img=left_image_msg, 
                                                          right_img=right_image_msg,
                                                          k_l=k_l,
                                                          k_r=k_r,
                                                          d_l=d_l,
                                                          d_r=d_r)
        
        # Step 2: Extract features
        l_key, l_des, r_key, r_des, l_key_dict = self.__extract_features__(left_image=rect_left, 
                                                                           right_image=rect_right)

        # Step 3: Stereo matching
        left_kp_to_3D = self._stereo_matching(l_key=l_key, 
                                              r_key=r_key, 
                                              l_des=l_des, 
                                              r_des=r_des,
                                              k_l=k_l)
        
        # If stereo matching fails, return
        if left_kp_to_3D is None:
            return

        # Case 1: If processing first image pair
        if self.first_frame is True:

            # Save extracted keypoints and descriptors
            self.prev_keypoints = l_key
            self.prev_descriptors = l_des
            self.prev_key_dict = l_key_dict

            # Save 2D-3D correspondences for left frame
            self.left_kp_to_3D_prev = left_kp_to_3D

            # Initialize back-end graph
            self._initialize_graph()

            # Mark that first image pair has been processed
            self.first_frame = False
            self.prev_time = curr_time

            self.prev_vel = np.array([[0.0], [0.0], [0.0]])

            return
        



        





        # Step 4: Temporal matching
        left_kp_prev_to_curr = self._temporal_matching(l_key=l_key, l_des=l_des, k_l=k_l)


        # Predict pose with IMU
        preintegrator = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)
        preintegrator, imu_pred_vel = self._predict_pose_from_imu(preintegrator=preintegrator, curr_time=curr_time)

        print("imu pred vel: ", imu_pred_vel)

        # Case 2: findEssentialMat fails
        if left_kp_prev_to_curr is None:
            return

        # Step 5: Pose estimation
        orientation, position = self.__estimate_pose_(left_kp_prev_to_curr=left_kp_prev_to_curr,
                                                      k_l=k_l,
                                                      curr_key_dict=l_key_dict)

        # If PnP fails, return
        if orientation is None:
            return
        
        # Predict pose with IMU
        preintegrator = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)
        preintegrator, imu_pred_vel = self._predict_pose_from_imu(preintegrator=preintegrator, curr_time=curr_time)

        # Graph optimization
        self._optimize_poses(orientation, position, preintegrator, imu_pred_vel)

        # Save previous keypoints and descriptors
        self.prev_keypoints = l_key
        self.prev_descriptors = l_des

        self.left_kp_to_3D_prev = left_kp_to_3D
        self.prev_key_dict = l_key_dict

        self.get_logger().info(f"finished processing frame {a}")

        return
    
    def imu_callback(self, msg: Imu):

        # Calculate time
        time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

        if self.prev_time is None:
            self.prev_time = time

        lin_accel = np.array([msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])
        ang_vel = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
        
        self.imu_buffer.append((time, lin_accel, ang_vel))

        return



























    ##################################################################
    #### Front end: feature matching, extraction, pose prediction ####
    ##################################################################

    def _image_preprocessing(self,
                             left_img,
                             right_img,
                             k_l: np.ndarray,
                             k_r: np.ndarray,
                             d_l: np.ndarray,
                             d_r: np.ndarray):

        # Convert images to CV-compatible data type
        left_img = self.bridge.imgmsg_to_cv2(img_msg=left_img, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(img_msg=right_img, desired_encoding='bgr8')

        # Retrieving image dimensions
        l_height, l_width = left_img.shape[:2]
        r_height, r_width = right_img.shape[:2]

        # Calculate rectification transformations
        R_l, R_r, P_l, P_r, Q, _, _  = cv.stereoRectify(cameraMatrix1=k_l,
                                                        distCoeffs1=d_l,
                                                        cameraMatrix2=k_r,
                                                        distCoeffs2=d_r,
                                                        imageSize=(l_width, l_height),
                                                        R=self.R,
                                                        T=self.T,
                                                        flags=cv.CALIB_ZERO_DISPARITY,
                                                        alpha=0)
        
        # Save projection matrices (for later use in triangulation)
        self.P_l = P_l
        self.P_r = P_r
        
        # Create maps to rectify images
        map_x_l, map_y_l = cv.initUndistortRectifyMap(cameraMatrix=k_l,
                                                      distCoeffs=d_l,
                                                      R=R_l,
                                                      newCameraMatrix=P_l,
                                                      size=(l_width, l_height),
                                                      m1type=cv.CV_32FC1)
        map_x_r, map_y_r = cv.initUndistortRectifyMap(cameraMatrix=k_r,
                                                      distCoeffs=d_r,
                                                      R=R_r,
                                                      newCameraMatrix=P_r,
                                                      size=(r_width, r_height),
                                                      m1type=cv.CV_32FC1)
        
        
        
        # Rectify left and right images
        rect_left  = cv.remap(left_img,  map_x_l, map_y_l, interpolation=cv.INTER_LINEAR)
        rect_right = cv.remap(right_img, map_x_r, map_y_r, interpolation=cv.INTER_LINEAR)

        return rect_left, rect_right


    def __extract_features__(self, left_image: np.ndarray, right_image: np.ndarray):

        # Extract features
        detector = self.detector
        l_key, l_des = detector.detectAndCompute(left_image, None)
        r_key, r_des = detector.detectAndCompute(right_image, None)

        l_key_dict = dict()
        r_key_dict = dict()

        # Add labels to keypoints
        for i, kp in enumerate(l_key):
            kp.class_id = i
            l_key_dict[i] = kp
        for i, kp in enumerate(r_key):
            kp.class_id = i
            r_key_dict[i] = kp

        return l_key, l_des, r_key, r_des, l_key_dict


    def _stereo_matching(self, l_key: list, r_key: list, l_des: list, r_des: list, k_l):

        # Brute-force matching of features
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        # Stereo matching
        matches_stereo = bf.knnMatch(l_des, r_des, k=2)

        # Lowe's ratio test
        good_matches = []
        for m,n in matches_stereo:
            if m.distance < 0.8*n.distance:
                good_matches.append(m)
        
        # Epipolar filtering
        good_matches_filtered = []
        for m in good_matches:
            uL, vL = l_key[m.queryIdx].pt
            uR, vR = r_key[m.trainIdx].pt

            if abs(vL-vR) <= 20:
                good_matches_filtered.append(m)

        good_matches = good_matches_filtered

        # Extract 2D coordinates of matches for left and right images
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        coords_left = np.array([l_key[m.queryIdx].pt for m in good_matches]).T
        coords_right = np.array([r_key[m.trainIdx].pt for m in good_matches]).T

        if coords_left.shape != coords_right.shape:
            self.get_logger().info("Stereo matching failed. Moving to next frame pair")
            return None

        # Estimate 3D points for matching pixels
        homogenous_coords_3d = cv.triangulatePoints(self.P_l, 
                                                    self.P_r, 
                                                    coords_left, 
                                                    coords_right)
        
        coords_3d = (homogenous_coords_3d[:3, :] / homogenous_coords_3d[3, :]).T

        # Mask to filter out points with negative depth + high reprojection error
        coords_left = coords_left.T

        # Reprojecting 3D estimates for filtering
        proj, _ = cv.projectPoints(coords_3d, np.zeros(3), np.zeros(3), k_l, None)
        proj = proj.reshape(-1,2)
        reproj_err = np.linalg.norm(proj - coords_left, axis=1)

        # mask = (reproj_err < 5) & (coords_3d[:, 2] > 0)
        # mask = (coords_3d[:, 2] > 0)
        # mask = (reproj_err < 5)
        mask = np.ones(len(coords_3d), dtype=bool)

        left_kp_positive = []
        for i, m in enumerate(good_matches):
            if mask[i]:
                left_kp_positive.append(l_key[m.queryIdx])

        # Filter corresponding 3D points
        positive_coords_3d = coords_3d[mask]
        
        # map from left kp to 3d
        left_kp_to_3D = dict()
        for kp, coord in zip(left_kp_positive, positive_coords_3d):
            left_kp_to_3D[kp.class_id] = coord
        
        return left_kp_to_3D

    def _temporal_matching(self, l_key, l_des, k_l):

        # Retrieve previous keypoints and descriptors
        prev_key = self.prev_keypoints
        prev_des = self.prev_descriptors

        # Brute-force matching of features
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        # Temporal matching
        matches_temporal = bf.knnMatch(prev_des, l_des, k=2)

        # Lowe's ratio test
        good_matches = []
        for m,n in matches_temporal:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        # Extract 2D coordinates of matches for left and right images
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        coords_curr = np.array([l_key[m.trainIdx].pt for m in good_matches])
        coords_prev = np.array([prev_key[m.queryIdx].pt for m in good_matches])
        kps_curr = np.array([l_key[m.trainIdx] for m in good_matches])
        kps_prev = np.array([prev_key[m.queryIdx] for m in good_matches])

        if len(good_matches) <= 4:
            self.get_logger().error("Not enough temporal matches found")
            return None

        elif coords_curr.shape != coords_prev.shape:
            self.get_logger().error("Temporal matching failed")
            return None
        
        _, mask = cv.findEssentialMat(coords_prev,
                                      coords_curr, 
                                      cameraMatrix=k_l, 
                                      method=cv.RANSAC, 
                                      prob=0.95, 
                                      threshold=5.0)
        
        if mask is None:
            return None

        kps_curr = kps_curr[mask.ravel() == 1]
        kps_prev = kps_prev[mask.ravel() == 1]

        left_kp_prev_to_curr = dict()
        for kp_prev, kp_curr in zip(kps_prev, kps_curr):
            left_kp_prev_to_curr[kp_prev.class_id] = kp_curr.class_id

        return left_kp_prev_to_curr

    def __estimate_pose_(self, left_kp_prev_to_curr: dict, k_l, curr_key_dict):

        # Retrieving maps needed for PnP
        prev_2D_to_prev_3D = self.left_kp_to_3D_prev
        prev_2D_to_curr_2D = left_kp_prev_to_curr

        # Initialize arrays to hold points in order
        points_3d = []
        points_2d = []
        for prev_kp_id, curr_kp_id in prev_2D_to_curr_2D.items():
            point_3d = prev_2D_to_prev_3D.get(prev_kp_id)

            if point_3d is not None:
                curr_key = curr_key_dict[curr_kp_id]
                point_2d = curr_key.pt

                points_2d.append(point_2d)
                points_3d.append(point_3d)

        # Skip PnP if not enough matches
        if len(points_2d) < 4:
            self.get_logger().error("Less than 4 points. PnP unable to solve. Skipping frame...")
            return None, None
        
        # Convert points into array for PnP
        points_2d_array = np.stack(points_2d).astype(np.float64)
        points_3d_array = np.stack(points_3d).astype(np.float64)

        retval, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=points_3d_array,
                                                        imagePoints=points_2d_array,
                                                        cameraMatrix=k_l,
                                                        distCoeffs=None,
                                                        reprojectionError=5.0,
                                                        confidence=0.99,
                                                        flags=cv.SOLVEPNP_ITERATIVE)
        
        # Failure conditions
        if retval is False:
            self.get_logger().error("PnP unable to solve. Skipping frame...")
            return None, None
        
        elif inliers is None:
            self.get_logger().error("No inliers found from PnP. Skipping frame...")
            return None, None
        
        elif len(inliers) <= 4:
            self.get_logger().error("Not enough inliers found from PnP. Skipping frame...")
            return None, None

        # 
        R, _ = cv.Rodrigues(rvec)
        orientation = R.T
        position = -R.T @ tvec

        #
        T_global = np.eye(4)
        T_global[:3,:3] = orientation
        T_global[:3,3] = position.flatten()
        self.t_global = T_global

        return orientation, position
    
    ###########################################
    ############### Inertial ##################
    ###########################################
    def _predict_pose_from_imu(self, preintegrator, curr_time):


        if curr_time is None:
            print("ASDFASDFSADF")

        prev_time = self.prev_time
        readings_in_window = []

        # Remove imu readings until reaching time window
        while self.imu_buffer and (self.imu_buffer[0])[0] < prev_time:
            print("test1")
            self.imu_buffer.popleft()
        
        # Save imu readings in time window
        while self.imu_buffer and (self.imu_buffer[0])[0] < curr_time:
            print("test2")
            readings_in_window.append(self.imu_buffer.popleft())
        

        for reading in readings_in_window:
            reading_time = reading[0]
            lin_accel = reading[1]
            ang_vel = reading[2]

            # Remove gravity from linear acceleration
            lin_accel[2] = lin_accel[2] - 9.81

            elapsed_time = reading_time-prev_time

            # print("elapsed time: ", reading_time-prev_time)
            print("lin accel ", lin_accel)
            print("ang_velo ", ang_vel)

            if elapsed_time > 0:
                preintegrator.integrateMeasurement(
                    lin_accel,
                    ang_vel,
                    elapsed_time
                )

            prev_time = reading_time
        
        # Retrieve previous velocity and bias estimate
        prev_vel = self.prev_vel
        prev_bias = gtsam.imuBias.ConstantBias()

        # Retrieve previous pose estimate
        quaternion = self.prev_orientation
        q = gtsam.Rot3.Quaternion(quaternion[0], 
                                    quaternion[1], 
                                    quaternion[2], 
                                    quaternion[3])
        prev_nav_state = gtsam.NavState(q, self.prev_position.reshape(3,1), prev_vel)

        # Predict pose and velo
        pred_state = preintegrator.predict(prev_nav_state, prev_bias)
        pred_pose = pred_state.pose()
        pred_vel  = pred_state.velocity()

        # Update previous values
        self.prev_vel = pred_vel
        self.prev_pose = pred_pose

        print("pred_vel: ", pred_vel)


        return preintegrator, pred_vel

    ###########################################
    #### Back end: pose graph optimization ####
    ###########################################

    def _initialize_graph(self):

        try:

            # Initialize local pose graph
            graph = gtsam.NonlinearFactorGraph()

            # Insert nodes (values to optimize)
            values = gtsam.Values()

            # Creating node for initial pose
            initial_pose_symbol = gtsam.symbol("x", self.frame_number)
            quaternion = self.initial_quaternion
            q = gtsam.Rot3.Quaternion(quaternion[0], 
                                    quaternion[1], 
                                    quaternion[2], 
                                    quaternion[3])
            initial_pose = gtsam.Pose3(q, self.initial_position)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            values.insert(initial_pose_symbol, initial_pose)

            # Creating node for initial velocity
            initial_velo = np.zeros(3)
            initial_velo_symbol = gtsam.symbol("v", self.frame_number)
            values.insert(initial_velo_symbol, initial_velo)

            # Creating node for initial bias
            initial_bias = gtsam.imuBias.ConstantBias()
            initial_bias_symbol = gtsam.symbol("b", self.frame_number)
            values.insert(initial_bias_symbol, initial_bias)

            # Adding starting position constraint
            graph.add(gtsam.PriorFactorPose3(initial_pose_symbol, initial_pose, prior_noise))

            # Adding starting velo constraint
            velo_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            graph.add(gtsam.PriorFactorVector(initial_velo_symbol, initial_velo, velo_noise))

            # Adding starting bias constraint
            bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            graph.add(gtsam.PriorFactorConstantBias(initial_bias_symbol, initial_bias, bias_noise))

            # Update global graph
            self.isam.update(graph, values)
            self.curr_estimate = self.isam.calculateEstimate()

            print(self.curr_estimate)

        except Exception as e:
            self.get_logger().info(str(e))

        return


    def _optimize_poses(self, orientation, position, preintegrator, imu_pred_vel):

        # Converting new pose estimate to gtsam-friendly data types
        R = gtsam.Rot3(orientation)
        t = np.array(position, dtype=np.float64).reshape(3,1)

        # Create new symbols
        pose_key = gtsam.symbol("x", self.frame_number)
        velo_key = gtsam.symbol("v", self.frame_number)
        bias_key = gtsam.symbol("b", self.frame_number)

        # Add new pose, velo, bias nodes 
        pose = gtsam.Pose3(R, t)
        velo = imu_pred_vel
        bias = gtsam.imuBias.ConstantBias()

        # Create a values object
        new_values = gtsam.Values()
        new_values.insert(pose_key, pose)
        new_values.insert(velo_key, velo)
        new_values.insert(bias_key, bias)

        # Retrieve previous symbols
        prev_pose_key = None
        prev_velo_key = None
        prev_bias_key = None

        keys_sorted = sorted(self.curr_estimate.keys(), 
                             key=lambda k: gtsam.symbolIndex(k),
                             reverse=True)
        
        i = -1
        while prev_pose_key is None or prev_velo_key is None or prev_bias_key is None:
            symbol_letter = chr(gtsam.Symbol(keys_sorted[i]).chr())
            symbol_number = gtsam.Symbol(keys_sorted[i]).index()

            print(symbol_letter)
            print(symbol_number)

            symbol = gtsam.symbol(symbol_letter, symbol_number)

            

            if symbol_letter == "x":
                prev_pose_key = symbol
            elif symbol_letter == "v":
                prev_velo_key = symbol
            else:
                prev_bias_key = symbol

            i = i-1

        # Create a new graph
        new_graph = gtsam.NonlinearFactorGraph()

        # Retrieve previous pose node
        prev_pose = self.curr_estimate.atPose3(prev_pose_key)
        T_rel = prev_pose.between(pose)
        noise = gtsam.noiseModel.Diagonal.Sigmas([0.1]*6)

        # Create edge between current and previous pose and add to new graph
        factor = gtsam.BetweenFactorPose3(prev_pose_key, pose_key, T_rel, noise)
        new_graph.add(factor)

        # Add new IMU factor to new graph
        imu_factor = gtsam.ImuFactor(prev_pose_key, 
                                     prev_velo_key,
                                     pose_key, 
                                     velo_key,
                                     prev_bias_key, 
                                     preintegrator)
        new_graph.add(imu_factor)

        # Add new bias factor to new graph
        bias_factor = gtsam.BetweenFactorConstantBias(prev_bias_key,
                                                      bias_key,
                                                      bias,
                                                      noise)
        new_graph.add(bias_factor)

        # print(new_graph)


        # Update and get optimized values
        self.isam.update(new_graph, new_values)
        self.curr_estimate = self.isam.calculateEstimate()
        # print(self.curr_estimate)

        # Get the optimized current pose
        optimized_pose = self.curr_estimate.atPose3(pose_key)
        optimized_R = optimized_pose.rotation().matrix()
        optimized_t = optimized_pose.translation()
        
        # Update global transformation with optimized values
        T_global = np.eye(4)
        T_global[:3,:3] = optimized_R
        T_global[:3,3] = optimized_t
        self.t_global = T_global
        
        # Update prev_position with optimized value for next motion validation
        self.prev_position = optimized_t.reshape(3,1)
        
        return
    






























    def _print_predictions(self):

        with open("output/pose_predictions.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["frame", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

            curr_estimate = self.curr_estimate
            symbols = curr_estimate.keys()

            for symbol in symbols:
                time = self.symbol_to_time.get(symbol)
                pose = curr_estimate.atPose3(symbol)

                transl = pose.translation()
                quat = pose.rotation().toQuaternion()

                writer.writerow([
                    time,
                    transl[0], transl[1], transl[2],
                    quat.x(), quat.y(), quat.z(), quat.w(),
                ])
        return
