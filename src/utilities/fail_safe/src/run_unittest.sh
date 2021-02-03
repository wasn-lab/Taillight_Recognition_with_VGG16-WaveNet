#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)
set -e
set -x
cd ${repo_dir}/src/utilities/fail_safe/src
python heartbeat_test.py
python rosbag_utils_test.py
python sb_param_utils_test.py
python vk221_3_test.py
python vk221_4_test.py
python rosbag_sender_test.py
python jira_utils_test.py
python issue_reporter_test.py
python sb_rosbag_sender_test.py
python rosbag_compressor_test.py
python load_monitor_test.py
python car_model_helper_test.py
python load_collector_test.py
