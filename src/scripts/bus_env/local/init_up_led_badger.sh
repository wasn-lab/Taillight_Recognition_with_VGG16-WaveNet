#!/bin/bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

until ls /home/localization/itriadv/devel/setup.bash
do
  echo "Wait auto mount operational"
  sleep 1
done

source /home/localization/itriadv/devel/setup.bash
roslaunch --wait powerled led_manager.launch
