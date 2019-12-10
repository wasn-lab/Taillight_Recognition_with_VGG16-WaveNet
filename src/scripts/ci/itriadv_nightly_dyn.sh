#!/bin/bash
set -e
readonly build_type="${build_type:-Release}"

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build
readonly devel_dir=devel

pushd $repo_dir

for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

catkin_make -DCATKIN_ENABLE_TESTING=1
source devel/setup.bash

# Run tests
catkin_make run_tests_car_model_rostest_test_car_b1_b1_drivenet_60.test

