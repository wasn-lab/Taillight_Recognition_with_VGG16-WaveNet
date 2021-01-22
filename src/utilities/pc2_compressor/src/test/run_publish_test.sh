#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir/build
source $repo_dir/devel/setup.bash

make -j lidar_raw_2020-03-10-10-48-39_41.bag
rostest pc2_compressor publish_test_cmpr.test

popd
