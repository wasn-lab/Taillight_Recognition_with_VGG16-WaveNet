#!/bin/bash
set -x

script_dir=$(dirname $0)
repo_dir=${script_dir}/../../../..

car_model=$(rosparam get /car_model)
if [[ -z "${car_model}" ]]; then
  car_model=C1
fi

pushd ${repo_dir}
catkin_make -DCAR_MODEL=${car_model} -DCATKIN_WHITELIST_PACKAGES="camera_grabber;camera_utils;lidar_location_send;control_checker;gnss_utility;lidarxyz2lla;from_dspace;to_dspace;geofence;trimble_grabber;geofence_map_pp;geofence_map_pp_filter;vehinfo_pub;geofence_pp;rad_grab;control;car_model;dl_data;msgs;autoware_control_msgs;autoware_lanelet2_msgs;autoware_perception_msgs;autoware_planning_msgs;autoware_system_msgs;autoware_vehicle_msgs;image_compressor;dspace_tx;flag_management;lateral_control;long_control;speed_profile;geofence_switch;brake_can_io;steer_can_io;xbywire_checker;plc_fatek"
popd
