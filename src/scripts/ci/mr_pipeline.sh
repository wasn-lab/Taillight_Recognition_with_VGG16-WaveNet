#!/bin/bash
set -x
set -e

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir
git fetch

set +e
git merge-base --is-ancestor remotes/origin/master HEAD
if [[ "$?" != "0" ]]; then
  echo "The merge request is not based the latest master. Please do "
  echo "  $ git fetch"
  echo "  $ git merge remotes/origin/master --no-ff"
  echo "  $ git push"
  echo "to trigger the merge request pipeline again."
  exit 1
fi
set -e

python src/scripts/ci/check_file_size.py
python src/scripts/ci/check_locked_file.py

readonly clean_build_status=$(python src/scripts/ci/decide_dirty_clean_build.py)
echo ${clean_build_status}
if [[ "${clean_build_status}" =~ "Clean build" ]]; then
  bash src/scripts/ci/module_build.sh
else
  catkin_make
fi

source devel/setup.bash
python src/scripts/ci/run_pylint.py

popd
