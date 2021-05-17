#! /bin/bash
set -x
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

readonly PWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /rosout	/cam/back_top_120/jpg
python scripts/wait_topic.py --topic-name /cam/front_top_close_120/jpg
python scripts/wait_topic.py --topic-name /cam/left_back_60/jpg
python scripts/wait_topic.py --topic-name /cam/left_front_60/jpg
python scripts/wait_topic.py --topic-name /cam/right_back_60/jpg
python scripts/wait_topic.py --topic-name /cam/right_front_60/jpg

cd $PWD

roslaunch --wait web_video_server web_video_server.launch
