#! /bin/bash

#source /home/localization/itriadv/devel/setup.bash
source /home/localization/tmp_lanelet2/itriadv/devel/setup.bash

export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

roslaunch planning_launch planning_all_launch.launch ORGS:=0 route_choose:=3 force_disable_avoidance:=false
