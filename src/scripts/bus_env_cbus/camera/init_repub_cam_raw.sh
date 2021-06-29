#! /bin/bash
source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.4

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /cam/back_top_120/raw
python scripts/wait_topic.py --topic-name /cam/front_bottom_60/raw
python scripts/wait_topic.py --topic-name /cam/front_top_close_120/raw
python scripts/wait_topic.py --topic-name /cam/front_top_far_30/raw
python scripts/wait_topic.py --topic-name /cam/left_back_60/raw
python scripts/wait_topic.py --topic-name /cam/left_front_60/raw
python scripts/wait_topic.py --topic-name /cam/right_back_60/raw
python scripts/wait_topic.py --topic-name /cam/right_front_60/raw

cd $CWD

roslaunch --wait camera_grabber repub_image_topic_tools.launch
