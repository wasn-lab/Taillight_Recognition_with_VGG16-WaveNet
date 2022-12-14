#!/bin/bash
set -x
set -e

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
# clean up the previous build.
for _dir in build devel; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

catkin_make \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_WHITELIST_PACKAGES="msgs;car_model;camera_grabber;camera_utils" \
    -DCAR_MODEL=C1 \

popd
