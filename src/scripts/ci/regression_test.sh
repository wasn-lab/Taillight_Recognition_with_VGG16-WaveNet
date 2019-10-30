#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
export ROS_HOME=$repo_dir
cd $repo_dir/build

# Let catkin_make builds only the target run_tests.
make camera_utils_test
../devel/lib/camera_utils/camera_utils_test
make parknet_test
../devel/lib/itri_parknet/parknet_test

cd -
