#! /bin/bash

source /home/local/itriadv/devel/setup.bash
#source /home/localization/tmp/itriadv/devel/setup.bash

export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

roslaunch map_pub map_pub.launch
