#!/bin/bash
set -x
source ~/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.4

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /cam/front_bottom_60/detect_image
python scripts/wait_topic.py --topic-name /cam/front_top_far_30/detect_image
python scripts/wait_topic.py --topic-name /cam/front_top_close_120/detect_image
python scripts/wait_topic.py --topic-name /cam/right_back_60/detect_image
python scripts/wait_topic.py --topic-name /cam/left_back_60/detect_image


cd $CWD

roslaunch --wait image_compressor cmpr_web_streaming.launch
