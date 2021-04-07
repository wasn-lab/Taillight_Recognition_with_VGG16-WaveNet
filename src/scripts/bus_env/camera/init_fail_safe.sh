#! /bin/bash

source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.2.4

roslaunch --wait fail_safe fail_safe.launch
