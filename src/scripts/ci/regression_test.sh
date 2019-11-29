#!/bin/bash
set -x
set -e

if [[ ! -z "$(pgrep rosmaster)" ]]; then
  echo "OK: rosmaster is running."
else
  rosmaster --core &
fi

readonly repo_dir=$(git rev-parse --show-toplevel)
export ROS_HOME=$repo_dir
cd $repo_dir/build

# Let catkin_make builds only the target run_tests.
make camera_utils_test
../devel/lib/camera_utils/camera_utils_test
make parknet_test
../devel/lib/itri_parknet/parknet_test
make car_model_test
../devel/lib/car_model/car_model_test

cd -
