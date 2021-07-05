#!/bin/bash
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir/build
source $repo_dir/devel/setup.bash

set -x
make lidar_raw_2020-12-28-16-53-14_21.bag
rostest pc_transform publish_test_lidars_transform.test

popd
