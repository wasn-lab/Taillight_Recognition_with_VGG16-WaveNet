#!/bin/bash
set -x

script_dir=$(dirname $0)
repo_dir=${script_dir}/../../../..

car_model=$(rosparam get /car_model)
if [[ -z "${car_model}" ]]; then
  car_model=C1
fi

pushd $repo_dir
catkin_make -DCAR_MODEL=${car_model} -DCATKIN_WHITELIST_PACKAGES="car_model;msgs;autoware_perception_msgs;camera_utils;camera_grabber;image_compressor;object_costmap_generator;drivenet_lib;drivenet;libs;alignment;fail_safe;hungarian;itri_tracking_2d;msg_replay;msg_recorder"
popd
