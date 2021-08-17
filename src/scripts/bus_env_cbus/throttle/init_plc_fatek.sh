#! /bin/bash
source /home/itri/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.7

roslaunch plc_fatek plc_fatek.launch
