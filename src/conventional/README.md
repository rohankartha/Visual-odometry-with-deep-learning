# Visual Odometry with Deep Learning



## Modules
`src/odometry`: ROS nodes implementing conventional stereo visual odometry, deep-learning enhanced odometry, and visual-inertial odometry

`src/utils`: ROS node to publish images and camera calibration information from bag files and an orchestration node

## How to run
1. Build Docker container with `docker-compose.yml`, `Dockerfile.base`, `Dockerfile.dev`
2. Open two terminals in the container
3. Run controller node like normal python programs (e.g. `python3 controller.py`)
4. Enter command `ros2 topic pub /user_start std_msgs/msg/String "{data: 'start'}" --once` to start the experiment

## Data

To run the experiment you need to download the euroc mav dataset from OpenVINS (the ros2bag) and make sure it has a `.db3` file-type and is located in the `src/conventional/data` directory.

Note: You may need to update your PYTHONPATH environment variable and 
