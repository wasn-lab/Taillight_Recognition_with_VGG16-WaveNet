#!/bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

readonly car_model=$(rosparam get /car_model)
if [[ "${car_model}" == "C2" ]]; then
  roslaunch --wait lidar c2.launch
elif [[ "${car_model}" == "C3" ]]; then
  roslaunch --wait lidar c3.launch
else
  roslaunch --wait lidar c1.launch
fi
