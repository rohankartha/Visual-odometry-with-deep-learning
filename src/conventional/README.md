# Visual Odometry with Deep Learning



## Modules
`src/odometry`: ROS nodes implementing conventional stereo visual odometry, deep-learning enhanced odometry, and visual-inertial odometry

`src/utils`: ROS node to publish images and camera calibration information from bag files and an orchestration node

## How to run
1. Build Docker container with `docker-compose.yml`, `Dockerfile.base`, `Dockerfile.dev`
2. Open two terminals in the container
3. Run controller node like normal python programs (e.g. `python3 controller.py`)
4. Enter command `ros2 topic pub /user_start std_msgs/msg/String "{data: 'start'}" --once` to start the experiment

Note: You may need to update your PYTHONPATH environment variable
