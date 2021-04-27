#!/bin/bash
set -x
sleep 10;
# Old setting, streamed images freeze after 8 seconds.
# firefox "https://service.itriadv.co:8784/Unit/DriverDashboard?URL=local&R=true"
chromium-browser --kiosk --app="https://service.itriadv.co:8784/Unit/DriverDashboard?plate=MOREA&URL=local" &
sleep 5 
python /usr/local/bin/move_window.py -m DP-5 -w "駕駛艙畫面"
