#!/bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.3

roslaunch --wait msg_replay replay_cmpr_pc2_to_camera.launch
