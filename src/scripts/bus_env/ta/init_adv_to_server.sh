#! /bin/bash
#source /home/nvidia/Control_team/subsystems_ccode/devel/setup.bash
source /home/nvidia/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.6
#/home/nvidia/Control_team/subsystems_ccode/devel/lib/adv_to_server/adv_to_server
/home/nvidia/itriadv/devel/lib/adv_to_server/adv_to_server -tcp_srv
