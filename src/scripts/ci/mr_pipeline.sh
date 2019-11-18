#!/bin/bash
set -x
set -e

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
git fetch

python src/scripts/ci/check_file_size.py

readonly output=$(python src/scripts/ci/check_package_change.py)
if [[ "${output}" =~ "package.xml" ]]; then
  echo "merge request has package.xml -> Clean build"
  bash src/scripts/ci/module_build.sh
else
  echo "merge request doest not have package.xml -> Dirty build"
  catkin_make
fi

popd
