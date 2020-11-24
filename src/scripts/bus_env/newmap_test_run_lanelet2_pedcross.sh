#! /bin/bash
echo starting programs........

gnome-terminal  -e 'bash -c "source ~/itriadv/devel/setup.bash;./init_roscore.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10"'
gnome-terminal  -e 'bash -c "source .bashrc;echo itri | sudo -S bash ip_forwarding.sh"'
sleep 3
gnome-terminal  -e "ssh -t camera 'echo itri | sudo -S bash ip_forwarding.sh'"
sleep 3
gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S route add -net 192.168.3.0 netmask 255.255.255.0 gw 192.168.2.1"'

sleep 3
gnome-terminal  -e 'bash -c "source ~/itriadv/devel/setup.bash;./init_lidar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_web_video_server.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_gui_gateway.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "~/run_sys_check_gateway.sh;exec bash"'

#sleep 3
#gnome-terminal  -e 'bash -c "./init_fusion.sh;exec bash"'
#sleep 3
#gnome-terminal  -e 'bash -c "./init_tracking_pp.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_3d_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_xyz2lla_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'bash -c "./init_obj_msg_converter_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_map_based_prediction_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_dynamic_object_vis_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_edge_detection.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t ta "source .bashrc;bash reset_time.sh"'
sleep 15
gnome-terminal  -e 'ssh -t ta "./init_control_new_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_radar.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t ta "./init_adv_to_server_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t local "./tmp_init_map_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "./tmp_init_localization_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'ssh -t local "./init_localization_supervision.sh;exec bash"'

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
gnome-terminal  -e 'ssh -t local "./tmp_init_planning_all_lanelet2.sh;exec bash"'

sleep 3
gnome-terminal  -e 'ssh -t ta "./init_camera.sh;exec bash"'

sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_2d_lanelet2.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_2d_lanelet2_left.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_2d_lanelet2_right.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_tracking_2d_lanelet2_fov30.sh;exec bash"'
sleep 3
gnome-terminal  -e 'bash -c "./init_pedcross.sh;exec bash"'

sleep 3
#gnome-terminal  -e "bash -c 'export ROS_MASTER_URI=http://192.168.1.3:11311;export ROS_IP=192.168.1.3;source /home/lidar/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"

gnome-terminal  -e "bash -c 'source /home/lidar/itriadv/devel/setup.bash;roslaunch opengl_test GUI_B_car.launch;exec bash'"

sleep 3
#gnome-terminal  -e "ssh -t camera 'export ROS_MASTER_URI=http://192.168.1.3:11311;export ROS_IP=192.168.3.10;source /home/camera/Documents/itriadv_d/devel/setup.bash;cd /home/camera/Documents/itriadv_d;./devel/lib/lightnet_tainan_new_layout/lightnet_tainan_new_layout_node;exec bash'"
#gnome-terminal  -e "ssh -t camera 'source /home/camera/Documents/itriadv_d/devel/setup.bash;cd /home/camera/Documents/itriadv_d;./devel/lib/lightnet_tainan_new_layout/lightnet_tainan_new_layout_node;exec bash'"

sleep 3
gnome-terminal  -e "ssh -t camera './init_camera.sh;exec bash'"

sleep 10
#export ROS_MASTER_URI=http://192.168.1.3:11311
#export ROS_IP=192.168.1.3
#source /home/lidar/workspace/itriadv/devel/setup.bash

source /home/lidar/itriadv/devel/setup.bash
roslaunch detection_viz itriadv_viz.launch rviz_config:="U3_b5_lanelet2_pedcross"

$SHELL

