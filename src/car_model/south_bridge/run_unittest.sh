#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)
set -e
set -x
pushd ${repo_dir}/src/car_model/south_bridge
python car_model_helper.py
popd
