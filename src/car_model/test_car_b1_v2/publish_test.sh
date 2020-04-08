#!/bin/bash
set -x
set -e
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 1
fi

if [[ ! -f src/sensing/itri_drivenet/drivenet/data/yolo/yolov3_b1-kINT8-batch3.engine ]]; then
  bash src/car_model/test_car_b1_v2/init_test_env.sh
fi

source devel/setup.bash

rostest car_model publish_test_drivenet_b1_v2.test
rostest car_model publish_test_tpp_b1_v2.test
