#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly darknet_dir=${repo_dir}/src/utilities/alexab_darknet
readonly drivenet_dir=${repo_dir}/src/sensing/itri_drivenet/drivenet
readonly cfg_file=${drivenet_dir}/data/yolo/yolov3.cfg
readonly data_file=${darknet_dir}/cfg/drivenet_fov60.data
readonly weights_file=${drivenet_dir}/data/yolo/yolov3_b1.weights
readonly jpg=/tmp/my/20190401145936_camera_frontcenter_000018009.png

${darknet_dir}/build/darknet detector test ${data_file} ${cfg_file} ${weights_file} ${jpg} -thresh 0.25
