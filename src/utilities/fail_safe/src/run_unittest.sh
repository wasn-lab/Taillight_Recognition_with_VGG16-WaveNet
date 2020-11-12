#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)
set -e
set -x
cd ${repo_dir}/src/utilities/fail_safe/src
python rosbag_utils_test.py
python sb_param_utils_test.py
python vk221_3_test.py
python vk221_4_test.py
