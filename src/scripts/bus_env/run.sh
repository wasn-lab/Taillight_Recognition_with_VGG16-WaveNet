#! /bin/bash
#gnome-terminal -e "bash -c 'ssh camera;roslist;$SHELL'" --tab
gnome-terminal  -e 'ssh -t lidar "source ~/itriadv/devel/setup.bash;./run_lidar.sh;exec bash"'
roslaunch sdb camera.launch
