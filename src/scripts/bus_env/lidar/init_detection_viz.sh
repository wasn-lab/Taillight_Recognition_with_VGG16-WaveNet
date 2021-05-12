#! /bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3
sleep 50
roslaunch detection_viz itriadv_viz.launch rviz_config:=\"U3_b5_lanelet2_pedcross\"
