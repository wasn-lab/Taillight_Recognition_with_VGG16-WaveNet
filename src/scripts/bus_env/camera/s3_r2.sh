#! /bin/bash
echo starting programs........
gnome-terminal  -e 'ssh -t lidar "source ~/itriadv/devel/setup.bash;./init_roscore.sh;exec bash"'

#gnome-terminal -e  'echo itri | sudo -S route add default gw 192.168.3.1'
#sleep 3
gnome-terminal  -e 'ssh -t lidar "source .bashrc;echo itri | sudo -S route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10"'
gnome-terminal  -e 'ssh -t lidar "source .bashrc;echo itri | sudo -S bash ip_forwarding.sh"'
sleep 3
gnome-terminal  -e "bash -c 'echo itri | sudo -S bash ip_forwarding.sh'"
#gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S ifconfig eth0:400 192.168.1.222"'
#sleep 3
#gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S ifconfig enp4s0 192.168.2.10"'
#sleep 3
gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S route add -net 192.168.3.0 netmask 255.255.255.0 gw 192.168.1.3"'
sleep 3

gnome-terminal  -e 'ssh -t lidar "source ~/itriadv/devel/setup.bash;./init_lidar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "~/run_web_video_server.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "~/run_gui_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "~/run_sys_check_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "./init_fusion.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "./init_tracking_pp.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t lidar "./init_edge_detection.sh;exec bash"'
sleep 3

gnome-terminal  -e 'ssh -t ta "source .bashrc;bash reset_time.sh"'
sleep 15
gnome-terminal  -e 'ssh -t ta "./init_control_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_radar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_mmtp_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_local_planning_new.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_adv_to_server.sh;exec bash"'

#sleep 3
#gnome-terminal  -e 'ssh -t ta "./to_dspace.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t local "./init_map.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "./init_localization.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "./init_localization_supervision.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t ta "./init_camera.sh;exec bash"'
sleep 3
gnome-terminal  -e "bash -c 'source /home/camera/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"
sleep 3
gnome-terminal  -e "bash -c 'export ROS_MASTER_URI=http://192.168.3.1:11311;export ROS_IP=192.168.3.10;source /home/camera/itriadv_d/devel/setup.bash;cd /home/camera/itriadv_d;devel/lib/lightnet_tainan/lightnet_tainan_node;exec bash'"

sleep 1
export ROS_MASTER_URI=http://192.168.3.1:11311
export ROS_IP=192.168.3.10
source /home/camera/itriadv/devel/setup.bash
roslaunch sdb camera.launch

$SHELL 




