#!/bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=127.0.0.1

roslaunch --wait msg_replay replay_lidar_front_top_localization_at_localization.launch
