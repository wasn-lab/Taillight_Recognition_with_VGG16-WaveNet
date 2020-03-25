#!/bin/bash
# For generating drivenet tensorRT engine files.
set -x

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi
source devel/setup.bash

find src/sensing/itri_drivenet -name "*.engine" -exec rm {} \;
roslaunch drivenet b1_v2_drivenet_group_a.launch > /dev/null 2>&1 &
roslaunch drivenet b1_v2_drivenet_group_b.launch > /dev/null 2>&1 &
# node c use the same engine as node b
#roslaunch drivenet b1_v2_drivenet_group_c.launch > /dev/null 2>&1 &

sleep 5m
killall roslaunch
find src/sensing/itri_drivenet -name "*.engine"
