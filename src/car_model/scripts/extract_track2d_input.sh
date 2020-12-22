#!/bin/sh
set -x
set -e
readonly topics="/CameraDetection/polygon /cam_obj/front_bottom_60"

rosbag record $topics -o tracked_2d --lz4
