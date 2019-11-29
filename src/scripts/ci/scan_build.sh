#!/bin/bash
set -x
set -e
# Use debug mode to silence unused variables.
readonly build_type="${build_type:-Debug}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build_scan_build
readonly devel_dir=devel_scan_build

if [[ -d /var/www/html/scan_build ]]; then
  readonly output_dir=/var/www/html/scan_build
else
  readonly output_dir=/tmp
fi

pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
blacklist="dl_data"
#blacklist="dl_data;ndt_gpu;convex_fusion;lidar;output_results_by_dbscan;lidar_squseg_inference;ouster_driver;velodyne_laserscan;velodyne;velodyne_msgs;velodyne_driver;velodyne_pointcloud;lidars_grabber;libs;lidars_preprocessing;localization"

scan-build -o ${output_dir} catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DSCAN_BUILD_MODE=1 \
    -DCATKIN_BLACKLIST_PACKAGES="$blacklist" \
    -j6

find ${output_dir} -type d -exec chmod 755 {} \;
find ${output_dir} -type f -exec chmod 644 {} \;
echo "Visit http://ci.itriadv.co/scan_build/ to see the html results (accessible in itri.org.tw only)."
popd
