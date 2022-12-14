#!/bin/bash
readonly enp0s31f6=$(ifconfig enp0s31f6)
echo ${enp0s31f6} | grep "78:d0:04:2a:ad:50" > /dev/null 2>&1
if [[ "$?" == "0" ]]; then
  echo "Do not call sdb-run-cbus-screen.sh in the lab environment."
  echo "Use sdb-run-cbus-lab-screen.sh instead"
  exit 0
fi

set -x
echo starting programs........

readonly cur_dir=$(dirname $(readlink -e $0))
readonly car_model=$(rosparam get /car_model)

if [[ -z "${car_model}" ]]; then
  echo "car model not set. Use default setting"
  rosparam set /south_bridge/vid Oq5YN1hgzAhA
  rosparam set /south_bridge/license_plate_number MOREA
  rosparam set /south_bridge/company_name itri
  rosparam set /car_model C1
fi

gnome-terminal -e "screen -c ${cur_dir}/lidar.screen"
gnome-terminal -e 'ssh -t local "screen -c /home/local/itriadv/src/scripts/bus_env_cbus/localization.screen"'
#the current user names of throttle and camera ipc are wrong 
##############################################################################################################
#gnome-terminal -e "ssh -t camera 'screen -c /home/camera/itriadv/src/scripts/bus_env_cbus/camera.screen'"
#gnome-terminal -e "ssh -t throttle 'screen -c /home/itri/itriadv/src/scripts/bus_env_cbus/throttle.screen'"
gnome-terminal -e "ssh -t camera 'screen -c ~/itriadv/src/scripts/bus_env_cbus/camera.screen'"
gnome-terminal -e "ssh -t throttle 'screen -c ~/itriadv/src/scripts/bus_env_cbus/throttle.screen'"
##############################################################################################################
sleep 30
gnome-terminal -e 'ssh -t xavier "screen -c /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier.screen"'
