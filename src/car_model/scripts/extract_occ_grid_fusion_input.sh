#!/bin/sh
set -x
set -e
readonly topics="/occupancy_grid /LidarDetection/grid /CameraDetection/occupancy_grid /PathPredictionOutput/grid /nav_path_astar_base_30 /current_pose"

rosbag record $topics -o occ_grid_fusion --lz4
