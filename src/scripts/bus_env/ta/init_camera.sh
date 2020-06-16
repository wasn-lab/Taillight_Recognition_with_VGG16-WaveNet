#! /bin/bash
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=/opt/ros/kinetic/lib:/usr/lib:/usr/lib/aarch64-linux-gnu:/usr/lib:/usr/lib/aarch64-linux-gnu

source /home/nvidia/.bashrc
#source /home/nvidia/0225/itriadv-drivenet/devel/setup.bash
source /home/nvidia/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.2.10
roslaunch camera_grabber b1.launch
