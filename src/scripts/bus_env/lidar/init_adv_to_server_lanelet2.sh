#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
/home/lidar/itriadv/devel/lib/adv_to_server/adv_to_server -newMap -no_can -udp_srv
#/home/nvidia/test_ws/devel/lib/adv_to_server/adv_to_server -tcp_srv -udp_srv -newMap
#roslaunch --wait adv_to_server adv_to_server.launch
