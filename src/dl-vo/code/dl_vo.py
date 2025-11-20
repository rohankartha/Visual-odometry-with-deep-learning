#!/usr/bin/env python

# Importing core ROS dependencies
import rclpy # module for ROS APIs
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup

# Importing ROS message dependencies
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Importing reference frame libraries

# Importing external dependencies
import cv2 as cv
from cv_bridge import CvBridge

import torch
import sys
sys.path.append('/opt/SuperGluePretrainedNetwork')
from models.matching import Matching
from models.utils import frame2tensor

import gtsam
import numpy as np
import yaml
from gtsam import symbol

import csv
from datetime import datetime

# Constants
NODE_NAME = "stereo_vo"
USE_SIM_TIME = True
QUEUE_SIZE = 2900
MAX_DELAY = 0.05

# Topics
IMAGE_TOPIC_LEFT = "/cam0/image_raw"
CAMERA_TOPIC_LEFT = "/cam0/camera_info"
IMAGE_TOPIC_RIGHT = "/cam1/image_raw"
CAMERA_TOPIC_RIGHT = "/cam1/camera_info"

class DLVO(Node):

    def __init__(self, config_path, node_name=NODE_NAME, context=None):
        
        # Initialize Node super class
        super().__init__(node_name, context=context)

        # Retrieve camera-camera transformation matrix
        with open(config_path, "r") as fp:
            config = yaml.safe_load(fp)
        imu_T_cam0 = np.array(config["cam0"]["T_imu_cam"]).reshape(4,4)
        imu_T_cam1 = np.array(config["cam1"]["T_imu_cam"]).reshape(4,4)
        
        # Get transform from cam0 to cam1
        # T_c0_c1 = T_c0_imu @ T_imu_c1 = inv(T_imu_c0) @ T_imu_c1
        cam0_T_cam1 = np.linalg.inv(imu_T_cam0) @ imu_T_cam1

        # stereoRectify expects the transform from the first camera to the second.
        # Let's use the inverse transform (cam1 to cam0) to be safe.
        cam1_T_cam0 = np.linalg.inv(cam0_T_cam1)
        cam1_R_cam0 = cam1_T_cam0[:3,:3]
        cam1_t_cam0 = cam1_T_cam0[:3,3].reshape(3,1)

        self.R = cam1_R_cam0
        self.T = cam1_t_cam0
        # print(f"Stereo baseline: {np.linalg.norm(self.T):.6f} m")

        # Retrieve camera sizes
        self.l_cam_size = np.array(config["cam0"]["resolution"])
        self.r_cam_size = np.array(config["cam0"]["resolution"])

        # Initialize qos profile
        qos = QoSProfile(depth=2900)

        # Create callback group
        self.reentrant_cb_group = ReentrantCallbackGroup()

        # Synchronous subscriptions to image + camera topics
        self._left_image_sync_sub = Subscriber(self, Image, IMAGE_TOPIC_LEFT, 
            qos_profile=qos, callback_group=self.reentrant_cb_group)
        self._left_camera_sync_sub = Subscriber(self, CameraInfo, CAMERA_TOPIC_LEFT, 
            qos_profile=qos, callback_group=self.reentrant_cb_group)
        self._right_image_sync_sub = Subscriber(self, Image, IMAGE_TOPIC_RIGHT, 
            qos_profile=qos, callback_group=self.reentrant_cb_group)
        self._right_camera_sync_sub = Subscriber(self, CameraInfo, CAMERA_TOPIC_RIGHT, 
            qos_profile=qos, callback_group=self.reentrant_cb_group)

        # Synchronize image, depth, and camera messages
        queue_size = QUEUE_SIZE
        max_delay = MAX_DELAY
        self.time_sync = ApproximateTimeSynchronizer(
            [self._left_image_sync_sub, self._left_camera_sync_sub, self._right_image_sync_sub, self._right_camera_sync_sub],
            queue_size=queue_size, slop=max_delay
        )

        # Create member variables to hold features from previous frames
        self.prev_keypoints = None
        self.prev_descriptors = None

        # 
        self.curr_coord_map_3d = None
        self.prev_coord_map_3d = None

        # Create member variables to hold global rotation + translation matrices
        self.R_global = None
        self.t_global = None

        self.counter = 0

        # Create graph for pose optimization
        self.isam = gtsam.ISAM2()

        # Initialize local pose graph
        self.graph = gtsam.NonlinearFactorGraph()
        initial_pose = gtsam.Pose3()
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        new_symbol = symbol("x", self.counter)
        self.graph.add(gtsam.PriorFactorPose3(new_symbol, initial_pose, prior_noise))
        self.counter = self.counter + 1

        # Insert values
        self.values = gtsam.Values()
        self.values.insert(new_symbol, initial_pose)

        # Update global graph
        self.isam.update(self.graph, self.values)

        self.bridge = CvBridge()
        # Initialize SuperPoint + SuperGlue
        self.device = 'cpu'
        
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        
        self.matching = Matching(config).eval()

        # Register callback for synchronized messages
        self.time_sync.registerCallback(self.synced_camera_callback)
        self.get_logger().info("Time synchronized callbacks registered")

        self.first_frame = True
        self.prev_position = None
        # From ground truth data, max vel is 0.0522m/frame
        self.max_velocity = 0.2  # meters per frame
        # Open output file for trajectory logging
        self.trajectory_file = open('output.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.trajectory_file)
        self.csv_writer.writerow(['timestamp', 'frame', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
        self.trajectory_file.flush()
        self.frame_count = 0

    def synced_camera_callback(self, left_image_msg, left_camera_msg, right_image_msg, right_camera_msg):
        self.frame_count += 1
        # Extract left camera intrinsic matrix
        k_l = np.array(left_camera_msg.k).reshape(3,3)
        k_r = np.array(right_camera_msg.k).reshape(3,3)
        d_l = np.array(left_camera_msg.d)
        d_r = np.array(right_camera_msg.d)

        # Step 1: Preprocess image
        rect_left, rect_right = self._image_preprocessing(left_img=left_image_msg, right_img=right_image_msg,k_l=k_l,k_r=k_r,d_l=d_l,d_r=d_r)

        # Step 2: Extract features
        l_key, l_des, r_key, r_des = self.__extract_features__(left_image=rect_left, right_image=rect_right)
        print(f"✓ Features extracted: left={len(l_key)}, right={len(r_key)}")

        # Step 3: Stereo matching
        coords_3d_map = self._stereo_matching(l_key=l_key, r_key=r_key, l_des=l_des, r_des=r_des)
        print(f"✓ Stereo matching: {len(coords_3d_map)} 3D points")

        self.curr_coord_map_3d = coords_3d_map

        # Step 4: Temporal matching
        coords_2d_map = self._temporal_matching(l_key=l_key, l_des=l_des, k_l=k_l)
        if coords_2d_map is None:
            print("⚠ First frame - temporal matching skipped")
            self.prev_coord_map_3d = self.curr_coord_map_3d  # ← MOVE THIS HERE!
            return

        print(f"✓ Temporal matching: {len(coords_2d_map)} 2D-2D matches")

        # Step 5: Pose estimation
        pose = self.__estimate_pose_(coords_2d_map=coords_2d_map, coords_3d_map=self.prev_coord_map_3d, k_l=k_l, d_l=d_l, l_key=l_key)

        self.prev_coord_map_3d = self.curr_coord_map_3d  # Also keep it here for successful frames

        if pose is None:
            print("⚠ Pose estimation failed")
            return
        else:
            orientation, position = pose
            print(f"✓ Pose estimated: position={position.flatten()}")

        self.__optimize_poses(orientation, position)

        return


    def _image_preprocessing(self, left_img, right_img, k_l, k_r, d_l, d_r):
        # Calculate rectification transformations
        R_l, R_r, P_l, P_r, Q, _, _  = cv.stereoRectify(cameraMatrix1=k_l, distCoeffs1=d_l, cameraMatrix2=k_r, distCoeffs2=d_r, imageSize=self.l_cam_size, R=self.R, T=self.T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

        # Save projection matrices (for later use in triangulation)
        self.P_l = P_l
        self.P_r = P_r
        
        # Create maps to rectify images
        map_x_l, map_y_l = cv.initUndistortRectifyMap(cameraMatrix=k_l,distCoeffs=d_l,R=R_l,newCameraMatrix=P_l,size=self.l_cam_size,m1type=cv.CV_32FC1)
        map_x_r, map_y_r = cv.initUndistortRectifyMap(cameraMatrix=k_r,distCoeffs=d_r,R=R_r,newCameraMatrix=P_r,size=self.l_cam_size,m1type=cv.CV_32FC1)
        
        # Convert images to CV-compatible data type
        left_img = self.bridge.imgmsg_to_cv2(img_msg=left_img, desired_encoding='mono8')
        right_img = self.bridge.imgmsg_to_cv2(img_msg=right_img, desired_encoding='mono8')
        
        # Rectify left and right images
        rect_left  = cv.remap(left_img,  map_x_l, map_y_l, interpolation=cv.INTER_LINEAR)
        rect_right = cv.remap(right_img, map_x_r, map_y_r, interpolation=cv.INTER_LINEAR)

        return rect_left, rect_right

    def __extract_features__(self, left_image, right_image):
        """Extract features using SuperPoint"""
        
        # Images are already mono8 from preprocessing - no conversion needed!
        
        with torch.no_grad():
            # Convert to tensors
            left_tensor = frame2tensor(left_image, self.device)
            right_tensor = frame2tensor(right_image, self.device)
            
            # Extract features with SuperPoint
            data_left = {'image': left_tensor}
            data_right = {'image': right_tensor}
            
            pred_left = self.matching.superpoint(data_left)
            pred_right = self.matching.superpoint(data_right)
            
            # Get keypoints and descriptors
            l_kpts = pred_left['keypoints'][0].cpu().numpy()
            l_desc = pred_left['descriptors'][0].cpu().numpy().T
            r_kpts = pred_right['keypoints'][0].cpu().numpy()
            r_desc = pred_right['descriptors'][0].cpu().numpy().T
        
        # Convert to OpenCV KeyPoint format
        l_key = [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in l_kpts]
        r_key = [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in r_kpts]
        
        # Assign IDs
        for i, key in enumerate(l_key):
            key.class_id = i
        
        return l_key, l_desc, r_key, r_desc


    def _stereo_matching(self, l_key: list, r_key: list, l_des: list, r_des: list):

        # Brute-force matching of features
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

        # Stereo matching
        matches_stereo = bf.knnMatch(l_des, r_des, k=2)

        # Lowe's ratio test
        good_matches = []
        for m,n in matches_stereo:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        # Extract 2D coordinates of matches for left and right images
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        coords_left = np.array([l_key[m.queryIdx].pt for m in good_matches]).T
        coords_right = np.array([r_key[m.trainIdx].pt for m in good_matches]).T

        # triangulate for t
        homogenous_coords_3d = cv.triangulatePoints(self.P_l, self.P_r, coords_left, coords_right)
        
        coords_3d = (homogenous_coords_3d[:3, :] / homogenous_coords_3d[3, :]).T

        # Filter out points with invalid depths
        depths = coords_3d[:, 2]
        valid_depth_mask = (depths > 0.1) & (depths < 50.0) # Only trust points between 10cm and 50m

        # Apply the mask to points and their corresponding matches
        coords_3d = coords_3d[valid_depth_mask]
        
        # We need to create a new list of good_matches that corresponds to the valid depths
        original_good_matches = good_matches
        good_matches = []
        for i, is_valid in enumerate(valid_depth_mask):
            if is_valid:
                good_matches.append(original_good_matches[i])
        
        print(f"  → Filtered depths: {len(good_matches)}/{len(valid_depth_mask)} valid points")

        # Assign ids to 3d points
        labeled_3d_points = dict()
        stereo_matched_keypoints = []
        stereo_matched_descriptors = []

        for i, m in enumerate(good_matches):
            kp = l_key[m.queryIdx]
            labeled_3d_points[kp.class_id] = coords_3d[i]
            stereo_matched_keypoints.append(kp)
            stereo_matched_descriptors.append(l_des[m.queryIdx])

        # Save 3d coord map
        self.curr_coord_map_3d = labeled_3d_points
        self.stereo_matched_keypoints = stereo_matched_keypoints
        self.stereo_matched_descriptors = np.array(stereo_matched_descriptors)

        return labeled_3d_points
    

    def _temporal_matching(self, l_key, l_des, k_l):
        """Temporal matching using SuperGlue for better feature correspondence across time"""
        
        # Skip first frame pair - store for next iteration
        if self.prev_keypoints is None:
            self.prev_keypoints = self.stereo_matched_keypoints
            self.prev_descriptors = self.stereo_matched_descriptors
            # Store numpy arrays of keypoints for SuperGlue
            self.prev_kpts_array = np.array([kp.pt for kp in self.stereo_matched_keypoints], dtype=np.float32)
            return None
        
        # Prepare current keypoints array
        curr_kpts_array = np.array([kp.pt for kp in l_key], dtype=np.float32)
        
        # Run SuperGlue matching
        with torch.no_grad():
            # Convert to torch tensors
            data = {
                'keypoints0': torch.from_numpy(self.prev_kpts_array).unsqueeze(0).float().to(self.device),
                'keypoints1': torch.from_numpy(curr_kpts_array).unsqueeze(0).float().to(self.device),
                'descriptors0': torch.from_numpy(self.prev_descriptors.T).unsqueeze(0).float().to(self.device),
                'descriptors1': torch.from_numpy(l_des.T).unsqueeze(0).float().to(self.device),
                'scores0': torch.ones(1, len(self.prev_keypoints)).float().to(self.device),
                'scores1': torch.ones(1, len(l_key)).float().to(self.device),
            }
            
            # Add image shapes for SuperGlue
            data['image0'] = torch.zeros(1, 1, self.l_cam_size[1], self.l_cam_size[0]).to(self.device)
            data['image1'] = torch.zeros(1, 1, self.l_cam_size[1], self.l_cam_size[0]).to(self.device)
            
            # Run SuperGlue
            pred = self.matching.superglue(data)
  
            # Extract matches
            matches = pred['matches0'][0].cpu().numpy()  # Shape: (N,)
            confidence = pred['matching_scores0'][0].cpu().numpy()
        # Filter valid matches
        valid_mask = matches > -1
        
        # Get matched coordinates
        prev_key = self.prev_keypoints
        
        coords_prev = []
        coords_curr = []
        matched_prev_keys = []
        matched_curr_keys = []
    
        for i, match_idx in enumerate(matches):
            if match_idx > -1:  # Valid match
                coords_prev.append(prev_key[i].pt)
                coords_curr.append(l_key[int(match_idx)].pt)
                matched_prev_keys.append(prev_key[i])
                matched_curr_keys.append(l_key[int(match_idx)])
        print(f"  ✓ Coordinates extracted: {len(coords_prev)} matches")

        coords_prev = np.array(coords_prev, dtype=np.float32)
        coords_curr = np.array(coords_curr, dtype=np.float32)
        
        # Apply RANSAC with Essential Matrix for additional filtering
        E, mask = cv.findEssentialMat(
            coords_curr, 
            coords_prev, 
            cameraMatrix=k_l, 
            method=cv.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )

        if np.sum(mask) < 10:  # Need at least 20 RANSAC inliers
            self.get_logger().warn(f'Insufficient matches after RANSAC: {np.sum(mask)}')
            self.prev_keypoints = self.stereo_matched_keypoints
            self.prev_descriptors = self.stereo_matched_descriptors
            self.prev_kpts_array = np.array([kp.pt for kp in self.stereo_matched_keypoints], dtype=np.float32)
            return None
 
        # Filter by RANSAC mask
        coords_prev = coords_prev[mask.ravel() == 1]
        coords_curr = coords_curr[mask.ravel() == 1]
        
        # Build 2D coordinate map with class IDs
        coords_2d_map = dict()
        
        mask_flat = mask.ravel()
        inlier_idx = 0
        
        # Iterate through matched keypoints
        # matched_prev_keys and matched_curr_keys have same length as mask_flat
        for idx in range(len(matched_prev_keys)):
            if mask_flat[idx] == 1:  # Passed RANSAC
                prev_kp = matched_prev_keys[idx]
                curr_kp = matched_curr_keys[idx]
                # Use previous keypoint ID to map to current 2D coordinate
                coords_2d_map[prev_kp.class_id] = coords_curr[inlier_idx]
                inlier_idx += 1
        print(f"  ✓ Coordinate map built: {len(coords_2d_map)} entries")

        # Save current frame data for next iteration
        self.prev_keypoints = self.stereo_matched_keypoints
        self.prev_descriptors = self.stereo_matched_descriptors
        self.prev_kpts_array = np.array([kp.pt for kp in self.stereo_matched_keypoints], dtype=np.float32)
        
        return coords_2d_map
        
    def __estimate_pose_(self, coords_2d_map: dict, coords_3d_map: dict, k_l, d_l, l_key):

        # Initialize arrays to hold points in order
        points_3d = []
        points_2d = []

        matching_ids = coords_2d_map.keys() & coords_3d_map.keys()
        print(f"  → Overlapping IDs: {len(matching_ids)}")
        for matching_id in matching_ids:
            point_2d = np.array(coords_2d_map[matching_id])
            point_3d = np.array(coords_3d_map[matching_id])
            points_2d.append(point_2d)
            points_3d.append(point_3d)
            
        if len(points_2d) < 4:  # Need at least 4 points for PnP
            self.get_logger().warn(f'Insufficient 3D-2D correspondences: {len(points_2d)}')
            return None
        points_2d_array = np.stack(points_2d)
        points_3d_array = np.stack(points_3d)

        retval, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints=points_3d_array, imagePoints=points_2d_array, cameraMatrix=k_l, distCoeffs=d_l, reprojectionError=8.0, confidence=0.999, flags=cv.SOLVEPNP_ITERATIVE)

        if retval and inliers is not None:
            # Check if enough inliers
            if len(inliers) < 5:  # Need at least 5 inliers
                self.get_logger().warn(f'Too few PnP inliers: {len(inliers)}')
                return None
            
            # Check for unreasonable motion (outlier detection)
            translation_magnitude = np.linalg.norm(tvec)
            if translation_magnitude > 10.0:  # Threshold in meters
                self.get_logger().warn(f'Unreasonable translation: {translation_magnitude:.2f}m')
                return None
        
        R, _ = cv.Rodrigues(rvec)
        orientation = R.T
        position = -R.T @ tvec

        if self.prev_position is not None:
            delta = np.linalg.norm(position - self.prev_position)
            if delta > self.max_velocity:
                self.get_logger().warn(f'Motion too large: {delta:.2f}m')
                return None
        self.prev_position = position
        
        # Update global transformation
        T_global = np.eye(4)
        T_global[:3,:3] = orientation
        T_global[:3,3] = position.flatten()
        self.t_global = T_global

        return orientation, position


    def __optimize_poses(self, orientation, position):
        # Converting new pose estimate to gtsam-friendly data types
        R = gtsam.Rot3(orientation)
        t = np.array(position, dtype=np.float64).reshape(3,1)
        
        # Add new node for new pose estimate
        curr_symbol = symbol("x", self.counter)
        new_pose = gtsam.Pose3(R, t)
        new_values = gtsam.Values()
        new_graph = gtsam.NonlinearFactorGraph()
        new_values.insert(curr_symbol, new_pose)

        # Calculate relative transformation between current + prev estimate
        prev_symbol = symbol("x", self.counter-1)
        T_prev = self.values.atPose3(prev_symbol)
        T_relative = T_prev.between(new_pose)

        # Insert new edge (constraint) into graph with REALISTIC noise
        noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        new_factor = gtsam.BetweenFactorPose3(prev_symbol, curr_symbol, T_relative, noise)
        new_graph.add(new_factor)

        # Update and get optimized values
        self.isam.update(new_graph, new_values)
        self.values = self.isam.calculateEstimate()  # Save optimized values!

        # Get the optimized current pose
        optimized_pose = self.values.atPose3(curr_symbol)
        optimized_R = optimized_pose.rotation().matrix()
        optimized_t = optimized_pose.translation()
        
        from scipy.spatial.transform import Rotation as R_scipy
        quat = R_scipy.from_matrix(optimized_R).as_quat()  # Returns [x, y, z, w]
                
        # Log to file instead of printing
        self.csv_writer.writerow([
            f"{datetime.now().timestamp():.6f}",
            self.frame_count,
            f"{optimized_t[0]:.6f}",
            f"{optimized_t[1]:.6f}",
            f"{optimized_t[2]:.6f}",
            f"{quat[3]:.6f}",
            f"{quat[0]:.6f}",
            f"{quat[1]:.6f}",
            f"{quat[2]:.6f}"
        ])
        self.trajectory_file.flush()
        
        # Update global transformation with optimized values
        T_global = np.eye(4)
        T_global[:3,:3] = optimized_R
        T_global[:3,3] = optimized_t
        self.t_global = T_global
        
        # Update prev_position with optimized value for next motion validation
        self.prev_position = optimized_t.reshape(3,1)
        
        self.counter = self.counter + 1
        return