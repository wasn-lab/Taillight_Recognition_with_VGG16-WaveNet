#!/bin/bash

source /home/local/itriadv/devel/setup.bash

export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

roslaunch --wait planning_launch planning_all_launch.launch ORGS:=0 route_choose:=02 force_disable_avoidance:=false disable_lane_event:=true
