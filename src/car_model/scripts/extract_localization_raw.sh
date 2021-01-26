#!/bin/sh
set -x
set -e
readonly topics="/LidarFrontTop/Localization /gnss_utm_data /veh_info"

rosbag record $topics -o localization_raw --lz4
