#!/bin/bash
set -x
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.3


readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_node.py --node-name /web_video_server
python scripts/wait_node.py --node-name /xwin_grabber_rviz

cd $CWD
opera "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&R=true" &
python /usr/local/bin/move_window.py -m DP-5 -w "駕駛艙畫面"
echo "run infinite loop to raise HMI to be the top window."
while true; do
  python /usr/local/bin/raise_window.py -w "駕駛艙畫面"
  sleep 5
  set +x
done
