#!/bin/sh
set -x
set -e
readonly topics="/LidarFrontTop /gnss2local_data /veh_info"

rosbag record $topics -o localization_raw --lz4
