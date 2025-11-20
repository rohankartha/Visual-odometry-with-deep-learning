`launch`: scripts to launch gazebo with specified world and robot
`custom_models`: contains custom gazebo environments and robot models




setup empty room with custom robot:

export GAZEBO_MODEL_PATH=~/catkin_ws/src/custom_models/models
ros2 launch src/custom_models/empty_room.launch.py
python3 src/experiments/vo_linear.py

export PYTHONPATH=$PYTHONPATH:~/catkin_ws/src


ros2 topic pub /user_start std_msgs/msg/String "{data: 'start'}" --once

# Echo cam0 image messages
ros2 topic echo /cam0/image_raw

# Echo cam1 image messages
ros2 topic echo /cam1/image_raw