#!/usr/bin/env python

# Importing core ROS dependencies
import rclpy # module for ROS APIs
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# Importing ROS message dependencies
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Importing external dependencies
import cv2 as cv
from cv_bridge import CvBridge
import gtsam
import numpy as np
import yaml
from gtsam import symbol, symbolIndex
import csv
from collections import deque
from scipy.spatial.transform import Rotation as R_scipy
from datetime import datetime

# Constants
NODE_NAME = "stereo_vo"
USE_SIM_TIME = True
QUEUE_SIZE = 2900
MAX_DELAY = 0.05
PROCESSING_RATE = 0.3

# Topics
IMAGE_TOPIC_LEFT = "/cam0/image_raw"
CAMERA_TOPIC_LEFT = "/cam0/camera_info"
IMAGE_TOPIC_RIGHT = "/cam1/image_raw"
CAMERA_TOPIC_RIGHT = "/cam1/camera_info"


class StereoVO(Node):

    ###################
    ### Constructor ###
    ###################

    def __init__(self, config_path, initial_position, initial_quaternion,
                 node_name=NODE_NAME, context=None):
        
        # Initialize Node super class
        super().__init__(node_name, context=context)

        # Retrieve camera-camera transformation matrix
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)
        imu_T_cam0 = np.array(config["cam0"]["T_imu_cam"]).reshape(4,4)
        imu_T_cam1 = np.array(config["cam1"]["T_imu_cam"]).reshape(4,4)

        cam0_T_cam1 = np.linalg.inv(imu_T_cam0) @ imu_T_cam1
        cam1_T_cam0 = np.linalg.inv(cam0_T_cam1)
        cam1_R_cam0 = cam1_T_cam0[:3,:3]
        cam1_t_cam0 = cam1_T_cam0[:3,3].reshape(3,1)

        # Decompose into translation and rotation matrices
        self.R = cam1_R_cam0
        self.T = cam1_t_cam0

        # Initialize qos profile
        qos = QoSProfile(depth=10)

        # Create callback group
        self_mutually_excl_cb_group = MutuallyExclusiveCallbackGroup()

        # Synchronous subscriptions to image + camera topics
        self._left_image_sync_sub = Subscriber(self, 
                                          Image, 
                                          IMAGE_TOPIC_LEFT, 
                                          qos_profile=qos, 
                                          callback_group=self_mutually_excl_cb_group)
        self._left_camera_sync_sub = Subscriber(self,
                                           CameraInfo,
                                           CAMERA_TOPIC_LEFT,
                                           qos_profile=qos,
                                           callback_group=self_mutually_excl_cb_group)
        self._right_image_sync_sub = Subscriber(self, 
                                          Image, 
                                          IMAGE_TOPIC_RIGHT, 
                                          qos_profile=qos, 
                                          callback_group=self_mutually_excl_cb_group)
        self._right_camera_sync_sub = Subscriber(self,
                                           CameraInfo,
                                           CAMERA_TOPIC_RIGHT,
                                           qos_profile=qos,
                                           callback_group=self_mutually_excl_cb_group)

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

        # Create graph for pose optimization
        self.isam = gtsam.ISAM2()

        self.bridge = CvBridge()
        self.detector = cv.ORB_create(nfeatures=4000)

        # Register callback for synchronized messages
        self.time_sync.registerCallback(self.synced_camera_callback)
        self.get_logger().info("Time synchronized callbacks registered")

        # Boolean to track whether or not frame being processed is the first
        self.first_frame = True

        # Open output file for trajectory logging
        self.trajectory_file = open('output.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.trajectory_file)
        self.csv_writer.writerow(['timestamp', 'frame', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
        self.trajectory_file.flush()

        # Member variables to store initial state
        self.initial_position = initial_position
        self.initial_quaternion = initial_quaternion
        self.frame_number = 0

        # Queue to hold synchronized sensor messages
        self.message_queue = deque()

        # Enables offline processing + computation due to high frequency of msgs
        timer_period = PROCESSING_RATE
        self.timer = self.create_timer(timer_period, self.process_image)
    
    ###################
    #### Callbacks ####
    ###################
    
    # Receives sensor messages and pushes to queue
    def synced_camera_callback(self, 
                               left_image_msg: Image, 
                               right_image_msg: Image, 
                               left_camera_msg: CameraInfo, 
                               right_camera_msg: CameraInfo):
        
        self.message_queue.append((left_image_msg, 
                                   left_camera_msg, 
                                   right_image_msg, 
                                   right_camera_msg))

    # Main function
    def process_image(self):
        
        if not self.message_queue:
            return

        left_image_msg, left_camera_msg, right_image_msg, right_camera_msg = self.message_queue.popleft()
        
        self.get_logger().info(f"Processing frame pair {self.frame_number}")
        self.frame_number = self.frame_number + 1

        timestamp = left_image_msg.header.stamp.sec + (left_image_msg.header.stamp.nanosec * 1e-9)
        
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

        # Step 2: Feature extraction
        kpL, desL, id_to_kp_L = self.__extract_features__(left_image=rect_left)
        
        # Step 3: Stereo matching
        kpL_to_3D = self._stereo_matching(kpL=kpL, desL=desL, right_image=rect_right)
        
        # If stereo matching fails, return
        if kpL_to_3D is None:
            return

        # Case 1: If processing first image pair
        if self.first_frame is True:

            # Save extracted keypoints, descriptors, left image
            self.kpL_prev = kpL
            self.desL_prev = desL
            self.id_to_kp_L_prev = id_to_kp_L
            self.prev_imageL = rect_left

            # Save 2D-3D correspondences for left frame
            self.kpL_to_3D_prev = kpL_to_3D

            # Initialize back-end graph
            self._initialize_graph()

            # Mark that first image pair has been processed
            self.first_frame = False

            return

        # Step 4: Temporal matching
        left_kp_prev_to_curr = self._temporal_matching(kpL=kpL, desL=desL, k_l=k_l)

        # Case 2: findEssentialMat fails
        if left_kp_prev_to_curr is None:
            return

        # Step 5: Pose estimation
        orientation, position = self.__estimate_pose_(left_kp_prev_to_curr=left_kp_prev_to_curr,
                                                      k_l=k_l,
                                                      id_to_kp_L=id_to_kp_L)

        # If PnP fails, return
        if orientation is None:
            return
        
        # Graph optimization
        self._optimize_poses(orientation, position)

        # Save previous keypoints and descriptors
        self.kpL_prev = kpL
        self.desL_prev = desL
        self.id_to_kp_L_prev = id_to_kp_L

        # Save 2D-3D correspondences for left frame
        self.kpL_to_3D_prev = kpL_to_3D
        self.prev_imageL = rect_left

        return

    ##################################################################
    #### Front end: feature matching, extraction, pose prediction ####
    ##################################################################

    # Rectify and undistort images
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

    # Extract features from left image
    def __extract_features__(self, left_image: np.ndarray):

        # Extract features
        detector = self.detector
        kpL, desL = detector.detectAndCompute(left_image, None)

        id_to_kp_L = dict()

        # Add labels to keypoints
        for i, kp in enumerate(kpL):
            kp.class_id = i
            id_to_kp_L[i] = kp

        return kpL, desL, id_to_kp_L
    
    # Matching between left and right images
    def _stereo_matching(self, kpL, desL, right_image):

        # Calculating right image keypoints and descriptors
        detector = self.detector
        kpR, desR = detector.compute(right_image, kpL) 

        # Matching left image and right image features
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches_stereo = bf.knnMatch(desL, desR, k=2)
        matches = [m for m, n in matches_stereo]

        # Epipolar filtering
        epipolar_matches = []
        for m in matches:
            uL, vL = kpL[m.queryIdx].pt
            uR, vR = kpR[m.trainIdx].pt

            if abs(vL-vR) <= 5:
                if abs(uL-uR) < 30:
                    epipolar_matches.append(m)
 
        # Extract 2D coordinates of matches for left and right images
        epipolar_matches = sorted(epipolar_matches, key=lambda x: x.distance)
        coords_left = np.array([kpL[m.queryIdx].pt for m in epipolar_matches])
        coords_right = np.array([kpR[m.trainIdx].pt for m in epipolar_matches])

        # triangulatePoints requires 2xn
        coords_left_T = coords_left.T
        coords_right_T = coords_right.T

        if coords_left.shape != coords_right.shape:
            self.get_logger().info("Stereo matching failed. Moving to next frame pair")
            return None

        # Camera projection matrices from rectification
        P_l = np.asarray(self.P_l, dtype=np.float64)
        P_r = np.asarray(self.P_r, dtype=np.float64)

        # Estimate 3D points for matching pixels
        homogenous_coords_3d = cv.triangulatePoints(P_l, 
                                                    P_r, 
                                                    coords_left_T, 
                                                    coords_right_T)
        
        # Extract 3D point coordinates from homogenous coordinates
        coords_3d = (homogenous_coords_3d[:3, :] / homogenous_coords_3d[3, :])
        coords_3d_T = coords_3d.T

        # Remove very large points
        max_allowed = 10.0
        mask_large_points = np.all((coords_3d_T > -1*max_allowed) & (coords_3d_T < max_allowed), axis=1)

        # Filter out negative depth
        mask_negative_depth = coords_3d_T[:,2] > 0
        mask = mask_large_points & mask_negative_depth
        coords_3d_filtered = coords_3d_T[mask]
        final_matches = [m for m, keep in zip(epipolar_matches, mask) if keep]

        # Map feature ids to 3D points
        left_kp_to_3D = dict()
        for m, point in zip(final_matches, coords_3d_filtered):
            left_kp_to_3D[kpL[m.queryIdx].class_id] = point

        return left_kp_to_3D

    # Matching between left frames at t and t-1
    def _temporal_matching(self, kpL, desL, k_l):

        # Brute-force matching of features
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        # Temporal matching
        kpL_prev = self.kpL_prev
        desL_prev = self.desL_prev
        matches_temporal = bf.knnMatch(desL_prev, desL, k=2)

        print(f"{len(matches_temporal)} initial temporal matches")

        # Lowe's ratio test
        matches = []
        for m,n in matches_temporal:
            if m.distance < 0.65*n.distance:
                matches.append(m)
        
        print(f"{len(matches)} final temporal matches")
        
        # Extract 2D coordinates of matches for left and right images
        matches = sorted(matches, key=lambda x: x.distance)
        coords_curr = np.array([kpL[m.trainIdx].pt for m in matches])
        coords_prev = np.array([kpL_prev[m.queryIdx].pt for m in matches])
        kps_curr = np.array([kpL[m.trainIdx] for m in matches])
        kps_prev = np.array([kpL_prev[m.queryIdx] for m in matches])

        # Failure conditions
        if len(matches) <= 4:
            self.get_logger().error("Not enough temporal matches found")
            return None

        elif coords_curr.shape != coords_prev.shape:
            self.get_logger().error("Temporal matching failed")
            return None
        
        # Filter out inaccurate temporal matches
        _, mask = cv.findEssentialMat(coords_prev,
                                      coords_curr, 
                                      cameraMatrix=k_l, 
                                      method=cv.RANSAC, 
                                      prob=0.99, 
                                      threshold=1.0)

        if mask is None:
            return None

        kps_curr = kps_curr[mask.ravel() == 1]
        kps_prev = kps_prev[mask.ravel() == 1]

        # Map previous image feature ids to current image
        left_kp_prev_to_curr = dict()
        for kp_prev, kp_curr in zip(kps_prev, kps_curr):
            left_kp_prev_to_curr[kp_prev.class_id] = kp_curr.class_id

        return left_kp_prev_to_curr

    # Estimate pose from outputs of temporal and stereo matching
    def __estimate_pose_(self, left_kp_prev_to_curr: dict, k_l, id_to_kp_L):

        # Retrieving maps needed for PnP
        prev_2D_to_prev_3D = self.kpL_to_3D_prev
        prev_2D_to_curr_2D = left_kp_prev_to_curr

        # Initialize arrays to hold points in order
        points_3d = []
        points_2d = []
        for prev_kp_id, curr_kp_id in prev_2D_to_curr_2D.items():
            point_3d = prev_2D_to_prev_3D.get(prev_kp_id)

            if point_3d is not None:
                curr_key = id_to_kp_L[curr_kp_id]
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

        # PnP
        retval, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=points_3d_array,
                                                        imagePoints=points_2d_array,
                                                        cameraMatrix=k_l,
                                                        distCoeffs=None,
                                                        reprojectionError=5.0,
                                                        confidence=0.95,
                                                        flags=cv.SOLVEPNP_EPNP)

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

        # Recovering orientation and position
        R, _ = cv.Rodrigues(rvec)
        orientation = R.T
        position = -R.T @ tvec

        # Calculating global pose
        T_global = np.eye(4)
        T_global[:3,:3] = orientation
        T_global[:3,3] = position.flatten()
        self.t_global = T_global

        return orientation, position

    ###########################################
    #### Back end: pose graph optimization ####
    ###########################################

    def _initialize_graph(self):

        try:

            # Initialize quaternion
            quaternion = self.initial_quaternion

            # Initial position and add constraint
            first_symbol = gtsam.symbol("x", self.frame_number)
            q = gtsam.Rot3.Quaternion(quaternion[0],
                                      quaternion[1],
                                      quaternion[2],
                                      quaternion[3])
            initial_pose = gtsam.Pose3(q, self.initial_position)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

            # Initialize local pose graph
            graph = gtsam.NonlinearFactorGraph()
            graph.add(gtsam.PriorFactorPose3(first_symbol, initial_pose, prior_noise))

            # Insert values
            values = gtsam.Values()
            values.insert(first_symbol, initial_pose)

            # Update global graph
            self.isam.update(graph, values)
            self.curr_estimate = self.isam.calculateEstimate()

        except Exception as e:
            self.get_logger().info(e)

        return

    # 
    def _optimize_poses(self, orientation, position):

        try:

            # Converting new pose estimate to gtsam-friendly data types
            R = gtsam.Rot3(orientation)
            t = np.array(position, dtype=np.float64).reshape(3,1)
            
            # Add new node for new pose estimate
            curr_symbol = symbol("x", self.frame_number)
            new_pose = gtsam.Pose3(R, t)
            new_values = gtsam.Values()
            new_graph = gtsam.NonlinearFactorGraph()
            new_values.insert(curr_symbol, new_pose)

            # Calculate relative transformation between current + prev estiamte
            keys_sorted = sorted(self.curr_estimate.keys(), key=lambda k: symbolIndex(k))
            prev_symbol = keys_sorted[-1]
            T_prev = self.curr_estimate.atPose3(prev_symbol)
            T_relative = T_prev.between(new_pose)

            # Insert new edge (constraint) into graph
            noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            new_factor = gtsam.BetweenFactorPose3(prev_symbol, curr_symbol, T_relative, noise)
            new_graph.add(new_factor)

            # Update graph estimates
            self.isam.update(new_graph, new_values)
            self.curr_estimate = self.isam.calculateEstimate()

             # Get the optimized current pose
            optimized_pose = self.curr_estimate.atPose3(curr_symbol)
            optimized_R = optimized_pose.rotation().matrix()
            optimized_t = optimized_pose.translation()
        
            # Calculate trajectory
            quat = R_scipy.from_matrix(optimized_R).as_quat()
            row = [datetime.now().timestamp(), *optimized_t, quat[3], *quat[:3]]
            self.csv_writer.writerow(row)

            if self.frame_number % 100 == 0:
                self.trajectory_file.flush()
            
            # Update global transformation with optimized values
            T_global = np.eye(4)
            T_global[:3,:3] = optimized_R
            T_global[:3,3] = optimized_t
            self.t_global = T_global
            
            # Update prev_position with optimized value for next motion validation
            self.prev_position = optimized_t.reshape(3,1)

        except Exception as e:
            self.get_logger().info(str(e))
        
        return
    