#!/bin/bash
set -x
set -e

export CC=clang
export CXX=clang++

export build_dir=build_clang
export devel_dir=devel_clang
readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

readonly clean_build_status=$(python src/scripts/ci/decide_dirty_clean_build.py)
echo ${clean_build_status}
if [[ "${clean_build_status}" =~ "Clean build" ]]; then
  bash src/scripts/ci/module_build_clang.sh
else
  set +e
  catkin_make --build ${build_dir} -DCATKIN_DEVEL_PREFIX=${devel_dir} ${EXTRA_CATKIN_ARGS}
  if [[ ! "$?" == "0" ]]; then
    set -e
    echo "Dirty build fails. Try again with clean build."
    bash src/scripts/ci/module_build_clang.sh
  fi
fi
set -e

popd
