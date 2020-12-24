#! /bin/bash
echo starting programs........

gnome-terminal  -e 'bash -c "source ~/itriadv/devel/setup.bash;/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_roscore.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10"'
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S bash /home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/ip_forwarding.sh"'
sleep 3
gnome-terminal  -e "ssh -t camera 'echo itri | sudo -S bash /home/camera/itriadv/src/scripts/bus_env_cbus/camera/ip_forwarding.sh'"
sleep 3
gnome-terminal  -e 'ssh -t xavier "source .bashrc;echo nvidia | sudo -S route add -net 192.168.3.0 netmask 255.255.255.0 gw 192.168.2.1"'

gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_mps.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_lidar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/run_web_video_server.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/run_gui_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/run_sys_check_gateway.sh;exec bash"'

sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_fail_safe.sh;exec bash"'

#sleep 3
#gnome-terminal  -e 'bash -c "./init_fusion.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'bash -c "./init_tracking_pp.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'bash -c "./init_tracking_pp_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_tracking_3d_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_xyz2lla_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_obj_msg_converter_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_map_based_prediction_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_dynamic_object_vis_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_edge_detection.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t xavier "source .bashrc;bash /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/reset_time.sh"'
sleep 15
gnome-terminal  -e 'ssh -t xavier "/home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_control_new_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t xavier "/home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_radar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_adv_to_server_lanelet2.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'ssh -c "./init_adv_to_server_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t local "/home/local/itriadv/src/scripts/bus_env_cbus/local/tmp_init_map_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "/home/local/itriadv/src/scripts/bus_env_cbus/local/tmp_init_localization_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "/home/local/itriadv/src/scripts/bus_env_cbus/local/init_localization_supervision.sh;exec bash"'

#sleep 3
#gnome-terminal  -e 'ssh -t local "./tmp_init_lanelet2map_lanelet2.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'ssh -t local "./tmp_init_vehicle_description_lanelet2.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'ssh -t local "./tmp_init_planning_initial_lanelet2.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'ssh -t local "./tmp_init_planning_lanelet2.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'ssh -t local "./tmp_init_mission_input_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t local "/home/local/itriadv/src/scripts/bus_env_cbus/local/tmp_init_planning_all_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t xavier "/home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_camera.sh;exec bash"'

sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_tracking_2d_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_tracking_2d_lanelet2_left.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_tracking_2d_lanelet2_right.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_tracking_2d_lanelet2_fov30.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "/home/lidar/itriadv/src/scripts/bus_env_cbus/lidar/init_pedcross.sh;exec bash"'

sleep 3
#gnome-terminal  -e "bash -c 'export ROS_MASTER_URI=http://192.168.1.3:11311;export ROS_IP=192.168.1.3;source /home/lidar/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"

gnome-terminal  -e "bash -c 'source /home/lidar/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"

sleep 3
#gnome-terminal  -e "ssh -t camera 'export ROS_MASTER_URI=http://192.168.1.3:11311;export ROS_IP=192.168.3.10;source /home/camera/Documents/itriadv_d/devel/setup.bash;cd /home/camera/Documents/itriadv_d;./devel/lib/lightnet_tainan_new_layout/lightnet_tainan_new_layout_node;exec bash'"
#gnome-terminal  -e "ssh -t camera 'source /home/camera/Documents/itriadv_d/devel/setup.bash;cd /home/camera/Documents/itriadv_d;./devel/lib/lightnet_tainan_new_layout/lightnet_tainan_new_layout_node;exec bash'"

sleep 3
gnome-terminal  -e "ssh -t camera '/home/camera/itriadv/src/scripts/bus_env_cbus/camera/init_camera.sh;exec bash'"

sleep 3
gnome-terminal  -e "ssh -t camera '/home/camera/itriadv/src/scripts/bus_env_cbus/camera/init_fail_safe.sh;exec bash'"

sleep 10
#export ROS_MASTER_URI=http://192.168.1.3:11311
#export ROS_IP=192.168.1.3
#source /home/lidar/workspace/itriadv/devel/setup.bash

source /home/lidar/itriadv/devel/setup.bash
roslaunch detection_viz itriadv_viz.launch rviz_config:="U3_b5_lanelet2_pedcross"

$SHELL

