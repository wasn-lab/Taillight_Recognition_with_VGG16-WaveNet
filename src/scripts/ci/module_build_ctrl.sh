#!/bin/bash
set -x
set -e
export CC=/usr/local/llvm-6.0.0/bin/clang
export CXX=/usr/local/llvm-6.0.0/bin/clang++

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build
readonly devel_dir=devel
pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
whitelist="vehinfo_pub;lidarxyz2lla;geofence;mm_tp;virbb_pub;control;trimble_gps_imu_pub;dspace_subscriber;geofence_pp;gui_publisher;rad_grab;lidar_location_send;adv_to_server;msgs;car_model"


catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_WHITELIST_PACKAGES="$whitelist"
popd

