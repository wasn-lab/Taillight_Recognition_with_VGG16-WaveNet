#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
#source /home/lidar/workspace/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

roslaunch --wait itri_tracking_3p tpp.launch input_source:=1 create_polygon_from_bbox:=True show_classid:=True drivable_area_filter:=True
