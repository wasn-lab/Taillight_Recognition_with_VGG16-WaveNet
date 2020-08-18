#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly bag_dir=${repo_dir}/src/bags
pushd $repo_dir
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 1
fi

if [[ ! -f ${bag_dir}/auto_record_2020-03-10-10-48-39_41.bag
  || ! -f ${bag_dir}/auto_record_2020-04-14-16-41-15_89.bag
  || ! -f ${bag_dir}/lidar_raw_2020-03-10-10-48-39_41.bag
  || ! -f ${bag_dir}/edge_detection_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/localization_raw_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/rad_grab_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/ukf_mm_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/target_planner_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/lidarxyz2lla_2020-04-13-17-45-48_0.bag
  || ! -f ${bag_dir}/auto_record_2020-06-19-16-26-18_1_filtered.bag
  || ! -f ${bag_dir}/auto_record_2020-08-04-10-15-25_4_filtered.bag ]]; then
  bash src/car_model/test_car_b1_v2/init_test_env.sh
fi

source devel/setup.bash

# cache *.engine for quick loading
for engine in `find src/sensing -name "*.engine"`; do
  cat $engine > /dev/null 2>&1
done

export LD_PRELOAD=/usr/local/lib/libopencv_core.so
rostest car_model publish_test_drivenet_b1_v2.test
#rostest car_model publish_test_tpp_b1_v2.test
rostest car_model publish_test_track2d_b1_v2.test
#rostest car_model publish_test_pedcross_b1_v2.test
rostest car_model publish_test_lidarnet_b1_v2.test
rostest car_model publish_test_edge_detection_b1_v2.test
rostest car_model publish_test_localization_b1_v2.test
rostest car_model publish_test_lidarxyz2lla_b1_v2.test
rostest car_model publish_test_rad_grab_b1_v2.test
rostest car_model publish_test_geofence_pp_b1_v2.test
rostest car_model publish_test_ukf_mm_b1_v2.test
rostest car_model publish_test_target_planner_b1_v2.test
rostest car_model publish_test_drivenet_b1_v2_sidecam_3dobj.test

popd
