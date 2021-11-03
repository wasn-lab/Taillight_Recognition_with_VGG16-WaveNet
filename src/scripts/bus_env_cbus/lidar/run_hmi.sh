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
  opera "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&plate=C2" &
elif [[ "${car_model}" == "C3" ]]; then
  opera "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&plate=C3" &
else
  opera "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&plate=MOREA" &
fi

python scripts/move_window.py -m DP-5 -w "駕駛艙畫面"
while true; do
  python scripts/raise_window.py -w "駕駛艙畫面"
  sleep 5
  set +x
done
