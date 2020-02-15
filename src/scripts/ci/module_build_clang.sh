#!/bin/bash
set -x
set -e

if [[ -f /usr/local/llvm-6.0.0/bin/clang ]]; then
  export CC=/usr/local/llvm-6.0.0/bin/clang
  export CXX=/usr/local/llvm-6.0.0/bin/clang++
else
  export CC=clang
  export CXX=clang++
fi

readonly build_type="${build_type:-Release}"
readonly install_prefix="${install_prefix:-/usr/local/itriadv}"
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

catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_INSTALL_PREFIX=${install_prefix} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ${EXTRA_CATKIN_ARGS}
popd

