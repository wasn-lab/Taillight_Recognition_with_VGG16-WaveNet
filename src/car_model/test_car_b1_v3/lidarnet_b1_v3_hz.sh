#!/bin/bash
set -x
set -e
readonly logf=/tmp/lidarnet_node_hz.log

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V3" ]]; then
  echo "This script is for B1_V3 only."
  exit 0
fi

set +x
source devel/setup.bash
set -x

roslaunch car_model lidarnet_b1_v3_hz.test | tee ${logf}

echo "SUMMARY: Performance result:"
grep " Hz: " ${logf} | grep -v "No data" | sort
