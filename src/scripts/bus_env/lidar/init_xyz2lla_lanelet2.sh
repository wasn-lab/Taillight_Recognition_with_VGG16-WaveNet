#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
#source /home/lidar/workspace/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

roslaunch itri_xyz2lla xyz2lla.launch
