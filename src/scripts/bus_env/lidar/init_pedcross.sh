#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
roslaunch --wait itri_pedcross ped.launch skip_frame_number:=4
