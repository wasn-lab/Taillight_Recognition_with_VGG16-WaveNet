#!/bin/sh
set -x
set -e
readonly topics="/current_pose"

rosbag record $topics -o lidarxyz2lla --lz4
