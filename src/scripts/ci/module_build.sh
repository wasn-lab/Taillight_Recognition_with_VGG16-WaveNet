#!/bin/bash
set -x
set -e

readonly build_type="${build_type:-Release}"
readonly install_prefix="${install_prefix:-/usr/local/itriadv}"
readonly repo_dir=$(git rev-parse --show-toplevel)
# clean up the previous build.
for _dir in build devel; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

if [[ $(hostname) == "ci" ]]; then
  EXTRA_CATKIN_ARGS="${EXTRA_CATKIN_ARGS} -j8"
fi

catkin_make \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_INSTALL_PREFIX=${install_prefix} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ${EXTRA_CATKIN_ARGS}
