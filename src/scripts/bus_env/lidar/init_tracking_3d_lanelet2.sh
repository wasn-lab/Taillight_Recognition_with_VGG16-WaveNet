#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
#source /home/lidar/workspace/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

roslaunch itri_tracking_3d tpp.launch show_classid:=True input_source:=6 create_bbox_from_polygon:=True
