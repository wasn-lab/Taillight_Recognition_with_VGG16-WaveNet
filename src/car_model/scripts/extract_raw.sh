#!/bin/sh
set -x
set -e
readonly topics="/LidarFrontLeft/Raw /LidarFrontRight/Raw /LidarFrontTop/Raw /cam/back_top_120 /cam/front_bottom_60 /cam/front_top_close_120 /cam/front_top_far_30 /cam/left_back_60 /cam/left_front_60 /cam/right_back_60 /cam/right_front_60 /AlphaBackLeft /AlphaBackRight /AlphaFrontLeft /AlphaFrontRight /AlphaSideLeft /AlphaSideRight /CubtekFront /DelphiFront /RadAlpha /RadFront /Flag_Info01 /Flag_Info02 /Flag_Info03 /veh_info"

rosbag record $topics -o sensor_raw --lz4
