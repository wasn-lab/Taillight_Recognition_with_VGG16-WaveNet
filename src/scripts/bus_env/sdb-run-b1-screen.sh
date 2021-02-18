#! /bin/bash
set -x
echo starting programs........

readonly cur_dir=$(dirname $(readlink -e $0))

gnome-terminal -e "screen -c ${cur_dir}/lidar.screen"
echo "Wait a few seconds before bringing up other nodes."
sleep 5;
#gnome-terminal -e 'ssh -t ta "screen -c /home/nvidia/itriadv/src/scripts/bus_env/ta.screen"'
gnome-terminal -e 'ssh -t xavier "screen -c /home/nvidia/itriadv/src/scripts/bus_env/xavier.screen"'
gnome-terminal -e 'ssh -t local "screen -c /home/localization/itriadv/src/scripts/bus_env/localization.screen"'
gnome-terminal -e "ssh -t camera 'screen -c /home/camera/itriadv/src/scripts/bus_env/camera.screen'"
