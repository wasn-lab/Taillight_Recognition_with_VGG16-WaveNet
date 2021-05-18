#! /bin/bash
set -x
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3

readonly CWD=`pwd`

roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_node.py --node-name /localization
python scripts/wait_node.py --node-name /lidar_point_pillars_car
python scripts/wait_node.py --node-name /lidar_point_pillars_ped_cyc
python scripts/wait_node.py --node-name /lidar_point_pillars_integrator

cd $CWD
roslaunch detection_viz itriadv_viz.launch rviz_config:="U3_b5_lanelet2_pedcross"
