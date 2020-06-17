#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
#roslaunch itri_tracking_pp tpp.launch show_absspeed:="true" pp_obj_min_kmph:="10.0"
roslaunch itri_tracking_pp tpp.launch occ_source:=1

