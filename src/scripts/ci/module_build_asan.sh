#!/bin/bash
# Experimental:
# Build executables with address sanitizer (ASAN)
# Use running cuda programs, use
# ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:alloc_dealloc_mismatch=0
set -x
set -e

readonly build_type="${build_type:-Debug}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build
readonly devel_dir=devel
#if [[ -d /usr/local/llvm-6.0.0/bin ]]; then
#  export PATH=/usr/local/llvm-6.0.0/bin:$PATH
#fi
#export CC=clang
#export CXX=clang++

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

catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DENABLE_ADDRESS_SANITIZER=1 \
    ${EXTRA_CATKIN_ARGS}
popd

