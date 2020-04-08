#!/bin/bash
set -x



source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi
source devel/setup.bash

# build necessary programs and download files
pushd build
set -e
make -j pcd_saver video_saver image_saver
make auto_record_2020-03-10-10-48-39_41.bag
make lidar_raw_2020-03-10-10-48-39_41.bag
set +e
popd

# Generatie drivenet tensorRT engine files.
find src/sensing/itri_drivenet -name "*.engine" -exec rm {} \;
roslaunch drivenet b1_v2_drivenet_group_a.launch > /dev/null 2>&1 &
roslaunch drivenet b1_v2_drivenet_group_b.launch > /dev/null 2>&1 &
# node c use the same engine as node b
#roslaunch drivenet b1_v2_drivenet_group_c.launch > /dev/null 2>&1 &

sleep 5m
killall roslaunch
find src/sensing/itri_drivenet -name "*.engine"
