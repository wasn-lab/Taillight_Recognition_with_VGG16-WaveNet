#!/bin/bash
set -x
script_dir=$(dirname $0)
repo_dir=${script_dir}/../../../..

car_model=$(rosparam get /car_model)
if [[ -z "${car_model}" ]]; then
  car_model=C1
fi

pushd $repo_dir
catkin_make -DCAR_MODEL=${car_model} -DCATKIN_WHITELIST_PACKAGES="decision_maker;lidar_location_send;control_checker;lidarxyz2lla;target_planner;from_dspace;ukf_mm;virbb_pub;to_dspace;geofence;occ_grid_lane;trimble_gps_imu_pub;geofence_map_pp;path_transfer;vehinfo_pub;geofence_pp;planning_initial;veh_predictwaypoint;gnss_utility;rad_grab;control;scene_register_checker;car_model;autoware_planning_rviz_plugin;autoware_vehicle_rviz_plugin;pose_history;osqp_interface;spline_interpolation;detection_viz;dl_data;map_launch;planning_launch;vehicle_launch;msgs;autoware_control_msgs;autoware_lanelet2_msgs;autoware_perception_msgs;autoware_planning_msgs;autoware_system_msgs;autoware_vehicle_msgs;mission_input;mission_planner;save_route;motion_velocity_optimizer;behavior_velocity_planner;lane_change_planner;turn_signal_decider;obstacle_avoidance_planner;obstacle_stop_planner;astar_search;costmap_generator;freespace_planner;scenario_selector;lanelet2_extension;map_loader;map_tf_generator;lanelet2_map_preprocessor;as;raw_vehicle_cmd_converter;camera_description;imu_description;additional_vehicle_info_generator;b_bus_description;localization;map_pub;ndt_gpu;cuda_downsample;localization_supervision;edge_detection;msg_replay;pc_transform;dspace_tx;flag_management;lateral_control;long_control;speed_profile;geofence_switch"
popd
