#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir/build
source $repo_dir/devel/setup.bash

make lidar_raw_2020-03-10-10-48-39_41_1s.bag
rostest pc2_compressor publish_test_cmpr.test

make lidar_compressed_2020-03-10-10-48-39_41.bag
rostest pc2_compressor publish_test_decmpr.test

popd
