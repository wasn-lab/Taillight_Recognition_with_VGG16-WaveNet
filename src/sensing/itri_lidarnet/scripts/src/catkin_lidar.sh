#!/bin/bash

# itriadv src path
read -p "Enter Path to itriadv: " path
cd "$path"
echo "itriadv path: ${path}"
echo "---------------------"
#------ pkg varible
velodyne_pkg="velodyne;velodyne_driver;velodyne_laserscan;velodyne_msgs;velodyne_pointcloud;"
ouster_pkg="ouster_client;ouster_ros;ouster_viz;"
depend_pkg="dl_data;car_model;msgs;lidar;libs;scripts;"

main_node="lidars_grabber;lidar_point_pillars;"
edge_node="edge_detection;"
local_node="localization;map_pub;ndt_gpu;cuda_downsample;"

ssn_node="lidars_preprocessing;lidar_squseg_inference;lidar_squseg_v2_inference;output_results_by_dbscan;"
compression_node="raw_points_processor;lidars_decoder;"

#------ main
PS3='Enter Car Type: '
options=("B1" "C1" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "B1")
            echo "--------> you chose B1"
	    sleep 1
        catkin_make -DCAR_MODEL=B1_V3 -DCATKIN_WHITELIST_PACKAGES="${velodyne_pkg}${ouster_pkg}${depend_pkg}${main_node}${edge_node}${local_node}"
	    echo "Build Done."
	    break
            ;;
        "C1")
            echo "--------> you chose C1"
	    sleep 1
	    catkin_make -DCAR_MODEL=C1 -DCATKIN_WHITELIST_PACKAGES="${velodyne_pkg}${ouster_pkg}${depend_pkg}${main_node}${edge_node}${local_node}"
	    echo "Build Done."
	    break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done








