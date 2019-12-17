#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

source devel/setup.bash

# Run tests
pushd build
make -j run_tests_car_model_rostest
echo "All done!"
popd

popd
