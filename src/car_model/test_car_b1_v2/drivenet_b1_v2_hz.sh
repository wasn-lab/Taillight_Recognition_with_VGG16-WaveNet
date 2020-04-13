#!/bin/bash
set -x
set -e
readonly logf_separately=/tmp/drivenet_node_hz.log
readonly logf_simultaneously=/tmp/drivenet_hz.log

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi

source devel/setup.bash

function run_nodes_one_by_one
{
true > $logf_separately

echo "STATUS: Run drivenet node a"
export dn_launch_file=b1_v2_drivenet_group_a.launch
roslaunch car_model drivenet_b1_v2_node_hz.test | tee -a ${logf_separately}

echo "STATUS: Run drivenet node b"
export dn_launch_file=b1_v2_drivenet_group_b.launch
roslaunch car_model drivenet_b1_v2_node_hz.test | tee -a ${logf_separately}

echo "STATUS: Run drivenet node c"
export dn_launch_file=b1_v2_drivenet_group_c.launch
roslaunch car_model drivenet_b1_v2_node_hz.test | tee -a ${logf_separately}
}

function run_all_nodes_simultaneously
{
true > $logf_simultaneously

echo "STATUS: Run all drivenet nodes simultaneously"
roslaunch car_model drivenet_b1_v2_hz.test | tee -a ${logf_simultaneously}
}

run_nodes_one_by_one
run_all_nodes_simultaneously

echo "SUMMARY: Performance of running each node separately:"
grep "detect_image Hz: " ${logf_separately} | grep -v "No data" | sort

echo "SUMMARY: Performance of running all nodes simultaneously:"
grep "detect_image Hz: " ${logf_simultaneously} | grep -v "No data" | sort
