#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
source /home/lidar/py3env/bin/activate
roslaunch fail_safe fail_safe.launch
