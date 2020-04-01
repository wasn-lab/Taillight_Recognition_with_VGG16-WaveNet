#!/bin/bash
# For generating drivenet tensorRT engine files.
set -x

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1" ]]; then
  echo "This script is for B1 only."
  exit 0
fi
source devel/setup.bash

find src/sensing/itri_drivenet -name "*.engine" -exec rm {} \;
roslaunch drivenet b1_drivenet_60.launch > /dev/null 2>&1 &
roslaunch drivenet b1_drivenet_120_1.launch > /dev/null 2>&1 &
roslaunch drivenet b1_drivenet_120_2.launch > /dev/null 2>&1 &

sleep 5m
killall roslaunch
find src/sensing/itri_drivenet -name "*.engine"
