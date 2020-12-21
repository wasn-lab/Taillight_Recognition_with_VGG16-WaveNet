#! /bin/bash

source /home/localization/itri_old_map/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.5

roslaunch localization_supervision localization_supervision.launch
