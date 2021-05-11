#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
#source /home/lidar/workspace/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

roslaunch --wait itri_tracking_3d tpp.launch input_source:=6 show_classid:=True drivable_area_filter:=False
