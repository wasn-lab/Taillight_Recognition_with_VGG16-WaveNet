#! /bin/bash
source /home/camera/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.4

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /cam_obj/front_bottom_60
python scripts/wait_topic.py --topic-name /cam_obj/front_top_far_30
python scripts/wait_topic.py --topic-name /cam_obj/front_top_close_120
python scripts/wait_topic.py --topic-name /cam_obj/back_top_120
python scripts/wait_topic.py --topic-name /cam_obj/left_back_60
python scripts/wait_topic.py --topic-name /cam_obj/left_front_60
python scripts/wait_topic.py --topic-name /cam_obj/right_back_60
python scripts/wait_topic.py --topic-name /cam_obj/right_front_60
python scripts/wait_topic.py --topic-name /LidarDetection

cd $CWD

roslaunch --wait alignment b1_v3_2d_3d_matching.launch
