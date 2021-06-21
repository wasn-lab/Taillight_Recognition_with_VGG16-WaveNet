#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir/build

make -j pc_transform_test
../devel/lib/pc2_compressor/pc_transform_test

popd
