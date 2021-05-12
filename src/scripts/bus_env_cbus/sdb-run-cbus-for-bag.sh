#! /bin/bash
set -x
echo starting programs........

readonly cur_dir=$(dirname $(readlink -e $0))
rosparam set /south_bridge/vid Oq5YN1hgzAhA
rosparam set /south_bridge/license_plate_number MOREA
rosparam set /south_bridge/company_name itri
rosparam set /car_model C1

# out-of-the-car setting, used for development.
rosparam set /fail_safe/should_post_issue 0

gnome-terminal -e "screen -c ${cur_dir}/lidar-for-bag.screen"

gnome-terminal -e 'ssh -t local "screen -c /home/local/itriadv/src/scripts/bus_env_cbus/localization.screen"'
gnome-terminal -e "ssh -t camera 'screen -c /home/camera/itriadv/src/scripts/bus_env_cbus/camera.screen'"

sleep 30
gnome-terminal -e 'ssh -t xavier "screen -c /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier-for-bag.screen"'
