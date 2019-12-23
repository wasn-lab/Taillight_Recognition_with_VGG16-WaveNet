#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly coverity_data_dir="coverity_data"
readonly coverity_root=$(readlink -e $(dirname $(which cov-build))/..)
pushd $repo_dir

# clean up the previous build.
for _dir in build devel ${coverity_data_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
# workaround for openroadnet
if [[ -d src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed ]]; then
  rm -rf src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed
fi
# blacklist="lidar;output_results_by_dbscan;lidar_squseg_inference;ouster_driver;velodyne_laserscan;velodyne;velodyne_msgs;velodyne_driver;velodyne_pointcloud;lidars_grabber;libs;lidars_preprocessing;dl_data"

whitelist="msgs;car_model;camera_utils;drivenet;drivenet_lib"
cov-build --dir ${coverity_data_dir} --emit-complementary-info \
catkin_make -DENABLE_CCACHE=0 -DCATKIN_WHITELIST_PACKAGES="$whitelist" ${EXTRA_CATKIN_ARGS}

for cfg in `ls ${coverity_root}/config/MISRA/*.config`; do
  cov-analyze --misra-config ${cfg} --dir coverity_data --strip-path ${repo_dir}/
done

# cov-commit-defects --host 140.96.109.174 --dataport 9090 --stream master --dir coverity_data --user ${USER_NAME} --password ${USER_PASSWORD}
echo "visit http://140.96.109.174:8080 to see the result (project: itriadv, stream: master)"

popd
