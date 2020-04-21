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

source devel/setup.bash

export TSAN_OPTIONS="second_deadlock_stack=1"
# Not all modules can run with thread sanitizer.
# rostest car_model publish_test_drivenet_b1_v2.test
rostest car_model publish_test_tpp_b1_v2.test
#rostest car_model publish_test_lidarnet_b1_v2.test
#rostest car_model publish_test_edge_detection_b1_v2.test
#rostest car_model publish_test_localization_b1_v2.test
rostest car_model publish_test_lidarxyz2lla_b1_v2.test
rostest car_model publish_test_rad_grab_b1_v2.test
rostest car_model publish_test_ukf_mm_b1_v2.test
rostest car_model publish_test_target_planner_b1_v2.test

popd
