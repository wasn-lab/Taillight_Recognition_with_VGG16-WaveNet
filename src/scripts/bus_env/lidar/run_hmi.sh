#!/bin/bash
sleep 10;
# Old setting, streamed images freeze after 8 seconds.
# firefox "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&R=true"
opera "http://service.itriadv.co:8785/Unit/DriverDashboard?URL=local&R=true" &
sleep 5 
python /usr/local/bin/move_window.py -m DP-5 -w "駕駛艙畫面"
echo "run infinite loop to raise HMI to be the top window."
while true; do
  python /usr/local/bin/raise_window.py -w "駕駛艙畫面"
  sleep 1
  set +x
done
