#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 1
fi

if [[ ! -f src/sensing/itri_drivenet/drivenet/data/yolo/yolov3_b1-kINT8-batch3.engine
  || ! -f src/bags/auto_record_2020-03-10-10-48-39_41.bag
  || ! -f src/bags/auto_record_2020-04-14-16-41-15_89.bag
  || ! -f src/bags/lidar_raw_2020-03-10-10-48-39_41.bag
  || ! -f src/bags/edge_detection_2020-04-13-17-45-48_0.bag
  || ! -f src/bags/localization_raw_2020-04-13-17-45-48_0.bag
  || ! -f src/bags/rad_grab_2020-04-13-17-45-48_0.bag
  || ! -f src/bags/ukf_mm_2020-04-13-17-45-48_0.bag
  || ! -f src/bags/target_planner_2020-04-13-17-45-48_0.bag
  || ! -f src/bags/lidarxyz2lla_2020-04-13-17-45-48_0.bag ]]; then
  bash src/car_model/test_car_b1_v2/init_test_env.sh
fi

source devel/setup.bash

rostest car_model publish_test_drivenet_b1_v2.test
rostest car_model publish_test_convex_fusion_b1_v2.test
rostest car_model publish_test_tpp_b1_v2.test
rostest car_model publish_test_track2d_b1_v2.test
rostest car_model publish_test_pedcross_b1_v2.test
rostest car_model publish_test_lidarnet_b1_v2.test
rostest car_model publish_test_edge_detection_b1_v2.test
rostest car_model publish_test_localization_b1_v2.test
rostest car_model publish_test_lidarxyz2lla_b1_v2.test
rostest car_model publish_test_rad_grab_b1_v2.test
rostest car_model publish_test_ukf_mm_b1_v2.test
rostest car_model publish_test_target_planner_b1_v2.test

popd
