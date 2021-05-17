#!/bin/bash
set -x
source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.4

readonly PWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /cam/left_back_60/raw
python scripts/wait_topic.py --topic-name /cam/left_front_60/raw
python scripts/wait_topic.py --topic-name /cam/right_back_60/raw
python scripts/wait_topic.py --topic-name /cam/right_front_60/raw

cd $PWD

roslaunch --wait drivenet b1_v3_drivenet_side.launch
