#!/bin/bash
set -x
echo starting programs........

readonly cur_dir=$(dirname $(readlink -e $0))
rosparam set /south_bridge/vid dc5360f91e74
rosparam set /south_bridge/license_plate_number 試0002
rosparam set /south_bridge/company_name itri
rosparam set /car_model B1

gnome-terminal -e "screen -c ${cur_dir}/lidar.screen"

gnome-terminal -e 'ssh -t local "screen -c /home/localization/itriadv/src/scripts/bus_env/localization.screen"'

sleep 30
gnome-terminal -e 'ssh -t xavier "screen -c /home/nvidia/itriadv/src/scripts/bus_env/xavier.screen"'
sleep 5
gnome-terminal -e "ssh -t camera 'screen -c /home/camera/itriadv/src/scripts/bus_env/camera.screen'"
