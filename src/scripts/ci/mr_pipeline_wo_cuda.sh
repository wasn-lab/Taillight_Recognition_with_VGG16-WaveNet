#!/bin/bash
set -x
set -e

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

echo "Put the merge request based on the tip code. If it fails, then the "
echo "developer must manually update the merge request."
git fetch
git merge remotes/origin/master --no-ff

git merge-base --is-ancestor remotes/origin/master HEAD
if [[ "$?" != "0" ]]; then
  echo "The merge request is not based the latest master. Please do "
  echo "  $ git fetch"
  echo "  $ git merge remotes/origin/master --no-ff"
  echo "  $ git push"
  echo "to trigger the merge request pipeline again."
  exit 1
fi

readonly merge_base=$(git merge-base origin/master HEAD)
readonly clean_build_status=$(python src/scripts/ci/decide_dirty_clean_build.py)
echo ${clean_build_status}
if [[ "${clean_build_status}" =~ "Clean build" ]]; then
  bash src/scripts/ci/module_build.sh
else
  set +e
  catkin_make
  if [[ ! "$?" == "0" ]]; then
    set -e
    echo "Dirty build fails. Try again with clean build."
    bash src/scripts/ci/module_build.sh
  fi
fi
set -e

set +x
popd
