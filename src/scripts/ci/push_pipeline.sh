#!/bin/bash
set -x
set -e

function join_by { local IFS="$1"; shift; echo "$*"; }

readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

src/scripts/ci/module_build.sh
src/scripts/ci/regression_test.sh
src/scripts/ci/module_build_clang.sh

popd
