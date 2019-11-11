#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

src/scripts/ci/module_build.sh
src/scripts/ci/regression_test.sh
src/scripts/ci/module_build_clang.sh

popd
