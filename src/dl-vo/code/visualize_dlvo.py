#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys

def integrate_trajectory(relative_data):
    """Accumulates relative poses (dx, dy, dz, d_quat) into a global trajectory."""
    # Initialize global pose (Start at 0,0,0 with identity rotation)
    current_position = np.zeros(3)
    current_rotation = R.from_matrix(np.eye(3))
    
    global_positions = [current_position.copy()]
        
    for index, row in relative_data.iterrows():
        # 1. Get relative translation and rotation from the row
        rel_trans = np.array([row['x'], row['y'], row['z']])
        # Scipy expects [x, y, z, w]
        rel_quat = [row['qx'], row['qy'], row['qz'], row['qw']] 
        rel_rot = R.from_quat(rel_quat)
        
        # 2. Update Global Position
        # Apply current rotation to the relative step so it points in the correct global direction
        step_global = current_rotation.apply(rel_trans)
        current_position = current_position + step_global
        
        # 3. Update Global Rotation
        current_rotation = current_rotation * rel_rot
        
        global_positions.append(current_position.copy())
        
    return np.array(global_positions)

def main():
    # 1. Load Data
    filename = 'output.csv'
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)

    # 2. Detect Data Type (Relative vs Global)
    # Calculate magnitude of the raw vectors in the file
    raw_magnitudes = np.linalg.norm(data[['x', 'y', 'z']].values, axis=1)
    avg_mag = np.mean(raw_magnitudes)

    global_positions, relative_motion_magnitudes = None, None

    # Threshold: If avg step is small (< 0.5m), it's likely frame-to-frame relative motion.
    # If avg position is large, it's likely global coordinates.
    if avg_mag < 0.5:
        print(f"Detected Data Type: RELATIVE MOTION (Avg step: {avg_mag:.4f}m)")
        
        # Calculate Global Path
        global_positions = integrate_trajectory(data)
        
        # Calculate Relative Motion array
        relative_motion_magnitudes = raw_magnitudes
        
    else:
        print(f"Detected Data Type: GLOBAL POSES (Avg mag: {avg_mag:.4f}m)")
        
        # Get Global Path
        global_positions = data[['x', 'y', 'z']].values
        
        # Calculate Relative Motion (Diff between consecutive global points)
        relative_motion_magnitudes = np.linalg.norm(np.diff(global_positions, axis=0), axis=1)

    # PLOTTING

    # Extract X, Y, Z for plotting
    x, y, z = global_positions[:, 0], global_positions[:, 1], global_positions[:, 2]

    # Plot 1: Frame-to-Frame Relative Motion (Velocity Profile)
    plt.figure(figsize=(10, 5))
    frames = np.arange(1, len(relative_motion_magnitudes) + 1)
    plt.plot(frames, relative_motion_magnitudes, marker='o', markersize=2, linestyle='-', color='b', linewidth=1)
    plt.xlabel('Frame')
    plt.ylabel('Relative Motion (m)')
    plt.title(f'Frame-to-Frame Relative Motion (Mean: {np.mean(relative_motion_magnitudes):.4f} m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('relative_motion_plot.png', dpi=300)
    print("Saved relative_motion_plot.png")

    # Plot 2: Top-Down Map (2D)
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, label='VO Trajectory', linewidth=1.5, color='blue')
    plt.scatter(x[0], y[0], color='green', marker='^', s=150, label='Start', zorder=5)
    plt.scatter(x[-1], y[-1], color='red', marker='x', s=150, label='End', zorder=5)
    plt.title("Top-Down Trajectory (Map View)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')  # Critical for valid map checking
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('trajectory_map_2d.png', dpi=300)
    print("Saved trajectory_map_2d.png")

    # Plot 3: 3D View
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='VO Trajectory', linewidth=1, color='blue')
    ax.scatter(x[0], y[0], z[0], color='green', marker='^', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=100, label='End')
    ax.set_title("3D Trajectory View")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig('trajectory_map_3d.png', dpi=300)
    print("Saved trajectory_map_3d.png")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()