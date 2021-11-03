#!/bin/bash
set -x

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V3" ]]; then
  echo "This script is for B1_V3 only."
  exit 0
fi
set +x
source devel/setup.bash
set -x

# build necessary programs and download files
pushd build
set -e
make -j pcd_saver video_saver image_saver
make camera_raw_2021-02-25-15-53-00_77.bag
make alignment_auto_record_2021-04-22-23-13-32_27.bag
make pedcross_2021-05-06-13-36-41_0_filtered.bag
make auto_record_2020-04-14-16-41-15_89.bag
make geofence_pp_2020-11-16-16-35-39.bag
make lidar_raw_compressed_2021-02-03.bag
make lidar_compressed_xyzir_2021-07-27-22-52-12_62.bag
make lidar_raw_2020-12-28-16-53-14_21.bag
make lidar_detection_car_ped_cyc_2020-12-28-16-53-14_21.bag
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
src/car_model/scripts/gen_drivenet_engine.py --package car_model --launch gen_drivenet_engine.launch

find src/sensing/itri_drivenet -name "*.engine"
