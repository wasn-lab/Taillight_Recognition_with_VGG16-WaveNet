#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.3.3

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_topic.py --topic-name /LidarFrontTop/Raw/xyzir
python scripts/wait_topic.py --topic-name /LidarFrontLeft/Raw/xyzir
python scripts/wait_topic.py --topic-name /LidarFrontRight/Raw/xyzir

cd $CWD

roslaunch --wait pc2_compressor cmpr.launch
