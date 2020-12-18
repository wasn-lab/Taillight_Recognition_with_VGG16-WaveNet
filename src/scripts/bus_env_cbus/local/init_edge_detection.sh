#! /bin/bash

source /home/localization/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

roslaunch edge_detection edge_detection.launch
