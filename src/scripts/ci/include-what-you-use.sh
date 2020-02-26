#!/bin/bash
set -x
set -e

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly iwyu_path=/usr/local/llvm-6.0.0/bin/include-what-you-use

if [[ ! -f ${iwyu_path} ]]; then
  echo "Cannot find iwyu binary. Exit."
  exit 0
fi

if [[ ! -d /usr/local/llvm-6.0.0/bin ]]; then
  echo "Cannot find clang 6. Exit."
  exit 0
fi

export PATH=/usr/local/llvm-6.0.0/bin:$PATH
export CC=clang
export CXX=clang++

# clean up the previous build.
for _dir in build devel; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

catkin_make \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=${iwyu_path} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ${EXTRA_CATKIN_ARGS}
