#! /bin/bash
source /home/nvidia/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.2.6
roslaunch --wait image_compressor cmpr.launch
