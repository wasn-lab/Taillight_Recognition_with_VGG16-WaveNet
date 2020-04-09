#!/bin/sh
set -x
set -e
readonly topics="/LidarFrontLeft/Raw /LidarFrontRight/Raw /LidarFrontTop/Raw"

rosbag record $topics -o lidar_raw --lz4
