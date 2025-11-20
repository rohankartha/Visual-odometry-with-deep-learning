#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GROUND_TRUTH_PATH = 'euroc_data/V1_01_easy/V1_01_easy.txt'

def main():
    try:
        # The EuRoC ground truth is a space-separated .txt file
        # Columns are: timestamp tx ty tz qx qy qz qw
        column_names = ['timestamp', 'p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w']
        
        # Read CSV with flexible whitespace separator
        gt_data = pd.read_csv(GROUND_TRUTH_PATH, sep=r'\s+', header=None, names=column_names, comment='#')

        # Extract position columns (p_x, p_y, p_z)
        positions = gt_data[['p_x', 'p_y', 'p_z']].values
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{GROUND_TRUTH_PATH}'")
        return
    except KeyError:
        print("Error: Could not find the required position columns (p_x, p_y, p_z).")
        return

    # --- PLOT 1: Relative Motion ---
    # Compute Euclidean distance between consecutive ground truth poses
    rel_motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    frames = np.arange(1, len(positions))

    plt.figure(figsize=(10, 5))
    plt.plot(frames, rel_motion, marker='o', markersize=2, linestyle='-', color='b', linewidth=1)
    plt.xlabel('Frame')
    plt.ylabel('Relative Motion (m)')
    plt.title(f'Frame-to-Frame Relative Motion (GT)\nMean: {np.mean(rel_motion):.4f} m/frame')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ground_truth_relative_motion_plot.png', dpi=300)
    print("Saved ground_truth_relative_motion_plot.png")

    # --- PLOT 2: Top-Down Map ---
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, label='GT Trajectory', linewidth=1.5, color='green')
    plt.scatter(x[0], y[0], color='blue', marker='^', s=150, label='Start', zorder=5)
    plt.scatter(x[-1], y[-1], color='red', marker='x', s=150, label='End', zorder=5)
    
    plt.title("Top-Down Trajectory (Ground Truth)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')  # This is crucial for accurate shape comparison
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ground_truth_trajectory_2d.png', dpi=300)
    print("Saved ground_truth_trajectory_2d.png")

    # --- PLOT 3: 3D View ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, label='GT Trajectory', linewidth=1, color='green')
    ax.scatter(x[0], y[0], z[0], color='blue', marker='^', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=100, label='End')
    
    ax.set_title("3D Trajectory View (Ground Truth)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig('ground_truth_trajectory_3d.png', dpi=300)
    print("Saved ground_truth_trajectory_3d.png")

    plt.show()

if __name__ == "__main__":
    main()