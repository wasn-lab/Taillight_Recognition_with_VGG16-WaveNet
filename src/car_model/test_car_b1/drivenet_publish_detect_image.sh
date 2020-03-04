#!/bin/bash
set -x
set -e
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1" ]]; then
  echo "This script is for B1 only."
  exit 0
fi

source devel/setup.bash
catkin_make run_tests_car_model_rostest_test_car_b1_drivenet_publish_detect_image.test

