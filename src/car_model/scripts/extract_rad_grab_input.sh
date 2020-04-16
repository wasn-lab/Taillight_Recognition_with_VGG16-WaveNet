#!/bin/sh
set -x
set -e
readonly topics="/RadFront"

rosbag record $topics -o rad_grab --lz4
