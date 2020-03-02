#!/bin/bash
set -x
set -e

readonly build_type="${build_type:-Release}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

readonly merge_base=$(git merge-base origin/master HEAD)
readonly affected_files=$(git diff --name-only ${merge_base})
for fname in ${affected_files}; do
  if [[ -f ${fname} ]]; then
    touch ${fname}
  fi
done

readonly clean_build_status=$(python src/scripts/ci/decide_dirty_clean_build.py)
echo ${clean_build_status}
if [[ "${clean_build_status}" =~ "Clean build" ]]; then
  bash src/scripts/ci/module_build_clang.sh
else
  set +e
  catkin_make --build build_clang -DCATKIN_DEVEL_PREFIX=devel_clang ${EXTRA_CATKIN_ARGS}
  if [[ ! "$?" == "0" ]]; then
    set -e
    echo "Dirty build fails. Try again with clean build."
    bash src/scripts/ci/module_build_clang.sh
  fi
fi
set -e

popd
