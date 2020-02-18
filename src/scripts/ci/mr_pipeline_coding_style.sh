#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)

if [[ ! ${repo_dir}/build_clang ]]; then
  echo "Cannot find build_clang directory. Run module_build_clang.sh first"
  exit 1
fi

pushd $repo_dir
python src/scripts/ci/check_global_var.py
python src/scripts/ci/check_misra_cpp2008_6_4_1.py

popd
