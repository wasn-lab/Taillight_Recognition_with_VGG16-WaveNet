#!/bin/bash

source ~/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.4

roslaunch --wait msg_recorder msg_recorder.launch
