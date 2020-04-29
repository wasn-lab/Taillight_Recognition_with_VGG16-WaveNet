#!/bin/sh
set -x
set -e
readonly topics="/current_pose /imu_data /veh_info"

rosbag record $topics -o ukf_mm --lz4
