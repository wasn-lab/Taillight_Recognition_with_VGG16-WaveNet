#! /bin/bash

source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.3.1:11311
export ROS_IP=192.168.3.10
roslaunch --wait itri_pedcross ped.launch skip_frame_number:=4
