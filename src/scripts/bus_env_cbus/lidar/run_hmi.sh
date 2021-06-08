#!/bin/bash
set -x

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_node.py --node-name /web_video_server
python scripts/wait_node.py --node-name /xwin_grabber_rviz
cd $CWD

opera "https://service.itriadv.co:8784/Unit/DriverDashboard?plate=MOREA&URL=local" &
python /usr/local/bin/move_window.py -m DP-5 -w "駕駛艙畫面"
while true; do
  python /usr/local/bin/raise_window.py -w "駕駛艙畫面"
  sleep 1
  set +x
done
