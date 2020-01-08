#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly coverity_data_dir="coverity_data"
readonly coverity_root=$(readlink -e $(dirname $(which cov-build))/..)
pushd $repo_dir

# clean up the previous build.
for _dir in build devel ${coverity_data_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

blacklist="lidar;output_results_by_dbscan;lidar_squseg_inference;ouster_driver;velodyne_laserscan;velodyne;velodyne_msgs;velodyne_driver;velodyne_pointcloud;lidars_grabber;libs;lidars_preprocessing;dl_data;opengl_test"
whitelist="itri_tracking_pp;itri_pedcross;drivenet;drivenet_lib;camera_utils;msgs;issue_reporter;msg_recorder;car_model;camera_grabber;itri_parknet;sdb;adv_to_server;detection_viz;sensor_fusion"
whitelist="${whitelist};libs_opn;openroadnet"
whitelist="${whitelist};opengl_test"
whitelist="${whitelist};control;dspace_subscriber;map_pub;trimble_gps_imu_pub;vehinfo_pub;mm_tp;virbb_pub;lidar_location_send;gui_publisher;geofence_pp;geofence;rad_grab;lidarxyz2lla"
whitelist="${whitelist};lidar;libs;output_results_by_dbscan;lidar_squseg_inference;lidars_preprocessing;scripts;lidars_grabber"
whitelist="${whitelist};edge_detection;cuda_downsample;approximate_progressive_morphological_filter"
#whitelist="${whitelist};localization;ndt_gpu;"

cov-build --dir ${coverity_data_dir} --emit-complementary-info \
catkin_make -DENABLE_CCACHE=1 -DCATKIN_WHITELIST_PACKAGES="$whitelist" ${EXTRA_CATKIN_ARGS}

cov-analyze -dir coverity_data --strip-path ${repo_dir}/
for cfg in `ls ${coverity_root}/config/MISRA/*.config`; do
  cov-analyze --disable-default --misra-config ${cfg} --dir coverity_data --strip-path ${repo_dir}/
done

# cov-commit-defects --host 140.96.109.174 --dataport 9090 --stream master --dir coverity_data --user ${USER_NAME} --password ${USER_PASSWORD}
#echo "visit http://140.96.109.174:8080 to see the result (project: itriadv, stream: master)"

popd
