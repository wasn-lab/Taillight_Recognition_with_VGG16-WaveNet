#!/bin/bash
set -x
set -e
readonly pcd_nums=2390
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly prog=${repo_dir}/devel/lib/pslot_detector/parknet_alignment_node

# Run this script in the build directory.
pushd ${repo_dir}/build
make parknet_alignment_node -j

function gen_json {
  output=$1
  cam_sn=$2
  $prog -pcd_nums $pcd_nums -output_filename $output -cam_sn $cam_sn
}

# play rosbag in another terminal
gen_json left120.json 4
gen_json front120.json 5
gen_json right120.json 6


popd
