#!/bin/sh
set -x
set -e
readonly topics="/LidarFrontLeft/Compressed /LidarFrontRight/Compressed /LidarFrontTop/Compressed /tf /tf_static"

rosbag record $topics -o lidar_raw --lz4
