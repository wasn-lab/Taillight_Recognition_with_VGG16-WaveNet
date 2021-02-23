#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

roslaunch --wait drivable_area_filter daf.launch input_source:=1 use_filter:=True
