#!/bin/bash
set -x

readonly CWD=`pwd`
roscd car_model
python scripts/check_ros_master.py --wait-until-alive
python scripts/wait_node.py --node-name /web_video_server
python scripts/wait_node.py --node-name /xwin_grabber_rviz
cd $CWD

readonly car_model=$(rosparam get /car_model)
if [[ "${car_model}" == "C2" ]]; then
  opera "http://service.itriadv.co:8785/Unit/DriverDashboard?plate=C2&URL=local" &
elif [[ "${car_model}" == "C3" ]]; then
  opera "http://service.itriadv.co:8785/Unit/DriverDashboard?plate=C3&URL=local" &
else
  opera "http://service.itriadv.co:8785/Unit/DriverDashboard?plate=MOREA&URL=local" &
fi

python scripts/move_window.py -m DP-5 -w "駕駛艙畫面"
while true; do
  python scripts/raise_window.py -w "駕駛艙畫面"
  sleep 5
  set +x
done
