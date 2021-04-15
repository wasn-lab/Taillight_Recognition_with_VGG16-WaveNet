#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd ${repo_dir}/src/utilities/powerled/src
python led_manager_test.py
popd
