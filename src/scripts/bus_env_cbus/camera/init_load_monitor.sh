#! /bin/bash
source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.2.1
roslaunch --wait fail_safe load_monitor.launch
