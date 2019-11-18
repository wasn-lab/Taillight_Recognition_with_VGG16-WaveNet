#!/bin/bash
set -x
set -e

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
git fetch

python src/scripts/ci/check_file_size.py

readonly xml_status=$(python src/scripts/ci/check_package_change.py)
if [[ "${xml_status}" =~ "package.xml" ]]; then
  echo "merge request has package.xml -> Clean build"
  bash src/scripts/ci/module_build.sh
  bash src/scripts/ci/module_build_clang.sh
else
  echo "merge request doest not have package.xml -> Dirty build"
  catkin_make
  catkin_make --build build_clang -DCATKIN_DEVEL_PREFIX=devel_clang
fi

popd
