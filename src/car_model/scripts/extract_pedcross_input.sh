#!/bin/sh
set -x
set -e
# /cam_obj/front_bottom_60 is used to generate /Tracking2D
readonly topics="/cam/front_bottom_60 /cam/front_bottom_60_crop /nav_path_astar_final /veh_info /cam_obj/front_bottom_60"

rosbag record $topics -o pedcross --lz4
