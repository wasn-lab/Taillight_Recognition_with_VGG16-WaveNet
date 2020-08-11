#!/bin/bash
# publish test in merge request pipeline.
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly bag_dir=${repo_dir}/src/bags
pushd $repo_dir
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi

if [[ ! -f ${bag_dir}/auto_record_2020-06-19-16-26-18_1_filtered.bag
  || ! -f ${bag_dir}/auto_record_2020-08-04-10-15-25_4.bag ]]; then
  bash src/car_model/test_car_b1_v2/init_test_env.sh
fi

set +x
source devel/setup.bash
set -x
# cache *.engine for quick loading
for engine in `find src/sensing -name "*.engine"`; do
  cat $engine > /dev/null 2>&1
done

if [[ -f /usr/local/lib/libopencv_core.so ]]; then
  # workaround for opencv 4.2/cv_bridge compatibility. Remove it in the future.
  export LD_PRELOAD=/usr/local/lib/libopencv_core.so
fi
rostest car_model publish_test_drivenet_b1_v2_sidecam_3dobj.test
rostest car_model publish_test_track2d_b1_v2.test

popd
