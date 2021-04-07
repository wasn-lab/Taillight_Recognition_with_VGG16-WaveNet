#!/bin/bash
source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.2.4

roslaunch --wait msg_replay replay_jpg_at_camera.launch
