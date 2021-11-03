#!/bin/bash
set -x
set -e

export CC=clang
export CXX=clang++

export build_dir=build_clang_c1
export devel_dir=devel_clang_c1
readonly build_type="${build_type:-Debug}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

if [[ $(hostname) == "ci" ]]; then
  EXTRA_CATKIN_ARGS="${EXTRA_CATKIN_ARGS} -j8"
fi

catkin_make --build ${build_dir} -DCATKIN_DEVEL_PREFIX=${devel_dir} \
  -DCAR_MODEL=C1 ${EXTRA_CATKIN_ARGS}

popd
