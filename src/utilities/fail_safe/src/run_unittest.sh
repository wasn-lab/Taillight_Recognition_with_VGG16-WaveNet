#!/bin/bash
set -e
set -x
python rosbag_utils_test.py
python sb_param_utils_test.py
