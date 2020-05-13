#! /bin/bash
REPO="~/itriadv/src/scripts/bus_env"
echo starting programs........
gnome-terminal  -e 'bash -c "source ~/itriadv/devel/setup.bash;./init_roscore.sh;exec bash"'
sleep 1
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10"'
sleep 1
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S bash ~/itriadv/src/scripts/bus_env/lidar/ip_forwarding.sh"'
sleep 3
gnome-terminal  -e "ssh -t camera 'echo itri | sudo -S bash ~/itriadv/src/scripts/bus_env/camera/ip_forwarding.sh'"
sleep 3
gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S route add -net 192.168.3.0 netmask 255.255.255.0 gw 192.168.1.3"'
sleep 3

gnome-terminal  -e 'bash -c "source ~/itriadv/devel/setup.bash;./init_lidar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_web_video_server.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_gui_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_sys_check_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_fusion.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_pp.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_edge_detection.sh;exec bash"'
sleep 3

gnome-terminal  -e 'ssh -t ta "source .bashrc;bash reset_time.sh"'
sleep 15
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_control_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_wayarea2grid.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_vector_map_loader.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_radar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_mmtp_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_local_planning_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_adv_to_server.sh;exec bash"'


sleep 3
gnome-terminal  -e 'ssh -t local "/home/localization/itriadv/src/scripts/bus_env/local/init_map.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "/home/localization/itriadv/src/scripts/bus_env/local/init_localization.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "/home/localization/itriadv/src/scripts/bus_env/local/init_localization_supervision.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t ta "/home/nvidia/itriadv/src/scripts/bus_env/ta/init_camera.sh;exec bash"'
sleep 3
gnome-terminal  -e "bash -c 'export ROS_MASTER_URI=http://192.168.3.1:11311;export ROS_IP=192.168.3.1;source /home/lidar/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"
sleep 3
gnome-terminal  -e "ssh -t camera 'export ROS_MASTER_URI=http://192.168.3.1:11311;export ROS_IP=192.168.3.10;source /home/camera/itriadv_d/devel/setup.bash;cd /home/camera/itriadv_d;devel/lib/lightnet_tainan/lightnet_tainan_node;exec bash'"
sleep 3
gnome-terminal  -e "ssh -t camera '/home/camera/itriadv/src/scripts/bus_env/camera/init_camera.sh;exec bash'"
sleep 1
export ROS_MASTER_URI=http://192.168.3.1:11311
export ROS_IP=192.168.3.1
source /home/lidar/itriadv/devel/setup.bash
roslaunch sdb gui.launch

$SHELL 




