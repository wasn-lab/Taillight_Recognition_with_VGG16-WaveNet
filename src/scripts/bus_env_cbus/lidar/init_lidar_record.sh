#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
roslaunch lidar b1.launch mps:=1 mode:=2
sleep 1 
#roslaunch lidar b1.launch hardware_enable:=0
#roslaunch sdb lidar.launch

