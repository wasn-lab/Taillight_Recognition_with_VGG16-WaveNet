#!/bin/bash

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V3" ]]; then
  echo "This script is for B1_V3 only."
  exit 0
fi
source devel/setup.bash
set -x

# build necessary programs and download files
pushd build
set -e
make -j pcd_saver video_saver image_saver
make auto_record_2020-03-10-10-48-39_41.bag
make auto_record_2020-04-14-16-41-15_89.bag
make geofence_pp_2020-11-16-16-35-39.bag
make lidar_raw_2020-03-10-10-48-39_41.bag
make lidar_raw_compressed_2021-01-12-16-01-42.bag
make edge_detection_2020-04-13-17-45-48_0.bag
make localization_raw_2020-09-24-17-02-06.bag
make lidarxyz2lla_2020-04-13-17-45-48_0.bag
make rad_grab_2020-04-13-17-45-48_0.bag
make ukf_mm_2020-04-13-17-45-48_0.bag
make target_planner_2020-04-13-17-45-48_0.bag
make auto_record_2020-06-19-16-26-18_1_filtered.bag
make tracking_2d_2020-11-16-15-02-12.bag
set +e
popd

# Generatie drivenet tensorRT engine files.
find src/sensing/itri_drivenet -name "*.engine" -exec rm {} \;
src/car_model/scripts/gen_drivenet_engine.py --package sdb --launch camera_b1_v3.launch

find src/sensing/itri_drivenet -name "*.engine"
