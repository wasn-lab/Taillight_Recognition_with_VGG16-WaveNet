#! /bin/bash

source /home/lidar/itriadv/devel/setup.bash
#source /home/lidar/tmp_ped/lanelet2/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
roslaunch --wait itri_pedcross_tf ped_tf.launch
