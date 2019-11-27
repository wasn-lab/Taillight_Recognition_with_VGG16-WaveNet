#!/bin/bash
set -x
set -e
readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build_scan_build
readonly devel_dir=devel_scan_build
readonly commit_id=$(git rev-parse HEAD)
readonly scan_build_result=/tmp/${commit_id}
pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir} ${scan_build_result}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
blacklist="dl_data"
scan-build -o ${scan_build_result} catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_BLACKLIST_PACKAGES="$blacklist"

#if [[ -d ${scan_build_result} ]]; then
#  pushd ${scan_build_result}
#  readonly _dirs=$(ls -d */)
#  for d in ${_dirs}; do
#    pushd $d
#    mv * ..
#    popd
#  done
#  popd
#fi

popd
