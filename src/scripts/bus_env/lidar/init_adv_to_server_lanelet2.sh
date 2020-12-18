#! /bin/bash
#source /home/nvidia/Control_team/subsystems_ccode/devel/setup.bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
#/home/nvidia/Control_team/subsystems_ccode/devel/lib/adv_to_server/adv_to_server
/home/lidar/itriadv/devel/lib/adv_to_server/adv_to_server -newMap -no_can
#/home/nvidia/itriadv/devel/lib/adv_to_server/adv_to_server
#/home/nvidia/test_ws/devel/lib/adv_to_server/adv_to_server -tcp_srv -udp_srv -newMap
