#!/bin/bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.6

until ls /home/nvidia/itriadv/devel/setup.bash
do
  echo "Wait auto mount operational"
  sleep 1
done

export HOSTNAME=`hostname`
source /home/nvidia/itriadv/devel/setup.bash
roslaunch --wait fail_safe load_monitor.launch
