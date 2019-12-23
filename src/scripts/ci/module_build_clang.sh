#!/bin/bash
set -x
set -e
export CC=clang
export CXX=clang++

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build_clang
readonly devel_dir=devel_clang
pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
# workaround for openroadnet
if [[ -d src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed ]]; then
  rm -rf src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed
fi
blacklist="ndt_gpu;convex_fusion;lidar;output_results_by_dbscan;lidar_squseg_inference;ouster_driver;velodyne_laserscan;velodyne;velodyne_msgs;velodyne_driver;velodyne_pointcloud;lidars_grabber;libs;lidars_preprocessing;dl_data;localization;libs_opn;openroadnet"

catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_BLACKLIST_PACKAGES="$blacklist" ${EXTRA_CATKIN_ARGS}
popd

