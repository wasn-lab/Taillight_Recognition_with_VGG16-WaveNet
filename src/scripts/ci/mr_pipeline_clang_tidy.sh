#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly merge_base=$(git merge-base origin/master HEAD)
readonly affected_files=$(git diff --name-only ${merge_base})

export PATH=/usr/local/llvm-6.0.0/bin:$PATH

if [[ ! ${repo_dir}/build_clang ]]; then
  echo "Cannot find build_clang directory. Run module_build_clang.sh first"
  exit 1
fi

pushd $repo_dir

for fname in $affected_files; do
  python run_clang_tidy.py --cpp $fname
done

popd
