#!/bin/bash
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly bag_dir=${repo_dir}/src/bags
pushd $repo_dir
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V3" ]]; then
  echo "This script is for B1_V3 only."
  exit 1
fi

if [[ ! -f ${bag_dir}/camera_raw_2021-02-25-15-53-00_77.bag
  || ! -f ${bag_dir}/auto_record_2020-11-04-15-29-32_0.bag
  || ! -f ${bag_dir}/auto_record_2020-04-14-16-41-15_89.bag
  || ! -f ${bag_dir}/geofence_pp_2020-11-16-16-35-39.bag
  || ! -f ${bag_dir}/lidar_raw_compressed_2021-02-03.bag
  || ! -f ${bag_dir}/lidar_detection_car_ped_cyc_2020-12-28-16-53-14_21.bag
  || ! -f ${bag_dir}/lidar_raw_2020-12-28-16-53-14_21.bag
  || ! -f ${bag_dir}/localization_raw_2020-09-24-17-02-06.bag
  || ! -f ${bag_dir}/lidarxyz2lla_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/rad_grab_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/ukf_mm_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/target_planner_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/auto_record_2020-06-19-16-26-18_1_filtered.bag
  || ! -f ${bag_dir}/tracking_2d_2020-11-16-15-02-12.bag ]]; then
  bash src/car_model/test_car_b1_v3/init_test_env.sh
fi

set +x
source devel/setup.bash
set -x

src/car_model/scripts/gen_drivenet_engine.py --package car_model --launch gen_drivenet_engine.launch
export LD_PRELOAD=/usr/local/lib/libopencv_core.so
rostest car_model publish_test_drivenet_b1_v3.test
rostest car_model publish_test_track2d_b1_v3.test
rostest car_model publish_test_pedcross_b1_v3.test
rostest car_model publish_test_lidarnet_b1_v3.test
rostest car_model publish_test_lidars_grabber_b1_v3.test
rostest car_model publish_test_lidarnet_b1_v3_raw_compressed.test
rostest car_model publish_test_lidar_point_pillars_integrator_b1_v3.test
rostest car_model publish_test_edge_detection_b1_v3.test
rostest car_model publish_test_localization_b1_v3.test
rostest car_model publish_test_lidarxyz2lla_b1_v3.test
rostest car_model publish_test_rad_grab_b1_v3.test
rostest car_model publish_test_geofence_pp_b1_v3.test
rostest car_model publish_test_ukf_mm_b1_v3.test
rostest car_model publish_test_target_planner_b1_v3.test

popd
