#! /bin/bash
set -x
echo starting programs........

readonly cur_dir=$(dirname $(readlink -e $0))

gnome-terminal -e "screen -c ${cur_dir}/lidar.screen"
sleep 5
gnome-terminal -e 'ssh -t xavier "screen -c /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier.screen"'
gnome-terminal -e 'ssh -t local "screen -c /home/local/itriadv/src/scripts/bus_env_cbus/localization.screen"'
gnome-terminal -e "ssh -t camera 'screen -c /home/camera/itriadv/src/scripts/bus_env_cbus/camera.screen'"
