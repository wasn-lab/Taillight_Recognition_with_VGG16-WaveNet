#!/bin/sh
set -x
set -e
readonly topics="/current_pose /rear_current_pose /ukf_mm_topic /nav_path_astar_final /veh_info"

rosbag record $topics -o target_planner --lz4
