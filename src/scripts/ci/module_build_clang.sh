#!/bin/bash
set -x
set -e
export CC=clang
export CXX=clang++

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

# clean up the previous build.
for _dir in build ; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
# workaround for openroadnet
if [[ -d src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed ]]; then
  rm -rf src/sensing/itri_openroadnet/libs_opn/TensorFlow/Installed
fi

catkin_make \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_BLACKLIST_PACKAGES="ndt_gpu;convex_fusion;lidar;output_results_by_dbscan;lidar_squseg_inference;ouster_driver;velodyne_laserscan;velodyne;velodyne_msgs;velodyne_driver;velodyne_pointcloud;lidars_grabber;libs;lidars_preprocessing"
popd
