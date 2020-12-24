#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

# set proper ros parameters about vid, license_plate_number etc
roscd car_model
cd south_bridge
python sb_set_ros_param.py

roslaunch fail_safe fail_safe.launch
