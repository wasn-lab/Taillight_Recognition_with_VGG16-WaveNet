#!/bin/bash
set -x
set -e

if [[ ! -z "$(pgrep rosmaster)" ]]; then
  echo "OK: rosmaster is running."
else
  echo "ERROR: rosmaster is not running. Contact system admin"
  exit 1
fi

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly car_model_h=car_model/include/car_model.h
export ROS_HOME=$repo_dir
pushd $repo_dir/build

# Let catkin_make builds only the target run_tests.
make car_model_test
../devel/lib/car_model/car_model_test

make camera_utils_test
../devel/lib/camera_utils/camera_utils_test

# Run tests that may be disabled by car_model
if grep -Fxq "#define ENABLE_PARKNET 1" ${car_model_h}
then
  make parknet_test
  ../devel/lib/itri_parknet/parknet_test
fi

popd

pushd $repo_dir
src/utilities/fail_safe/src/run_unittest.sh
popd

echo "ALL done!"
exit 0
