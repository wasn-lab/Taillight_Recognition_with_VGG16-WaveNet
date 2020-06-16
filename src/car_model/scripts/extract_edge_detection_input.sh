#!/bin/sh
set -x
set -e
readonly topics="/LidarAll /LidarFrontTop /LidarFrontRight /LidarFrontLeft"

rosbag record $topics -o edge_detection --lz4
