#!/bin/bash
set -x
set -e

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi

source devel/setup.bash
roslaunch car_model drivenet_b1_v2_node_b_hz.test
