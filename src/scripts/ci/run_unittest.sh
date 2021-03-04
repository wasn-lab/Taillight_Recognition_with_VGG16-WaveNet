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
export ROS_HOME=$repo_dir
pushd $repo_dir/build

# Let catkin_make builds only the target run_tests.
make -j car_model_test camera_utils_test lidar_test
export LD_PRELOAD=/usr/local/lib/libopencv_core.so
../devel/lib/car_model/car_model_test
../devel/lib/camera_utils/camera_utils_test
../devel/lib/libs/lidar_test

pushd $repo_dir
set +x
source devel/setup.bash
set -x
src/car_model/south_bridge/run_unittest.sh
src/utilities/fail_safe/src/run_unittest.sh
src/utilities/image_compressor/src/test/run_unittest.sh
src/utilities/image_compressor/src/test/run_publish_test.sh
src/utilities/pc2_compressor/src/test/run_unittest.sh
src/utilities/pc2_compressor/src/test/run_publish_test.sh
popd

echo "ALL done!"
exit 0
