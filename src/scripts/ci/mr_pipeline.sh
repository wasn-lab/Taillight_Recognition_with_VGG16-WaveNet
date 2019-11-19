#!/bin/bash
set -x
set -e

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
git fetch

python src/scripts/ci/check_file_size.py
python src/scripts/ci/check_locked_file.py

readonly clean_build_status=$(python src/scripts/ci/decide_dirty_clean_build.py)
echo ${clean_build_status}
if [[ "${clean_build_status}" =~ "Clean build" ]]; then
  bash src/scripts/ci/module_build.sh
  bash src/scripts/ci/module_build_clang.sh
else
  catkin_make
  catkin_make --build build_clang -DCATKIN_DEVEL_PREFIX=devel_clang
fi

popd
