#! /bin/bash
source /home/nvidia/local_planning/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.6
roslaunch waypoint_planner astar_avoid.launch
