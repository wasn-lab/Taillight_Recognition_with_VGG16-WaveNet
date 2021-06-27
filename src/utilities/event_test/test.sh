#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)

pushd $repo_dir
set +x
source devel/setup.bash
set -x

rostest event_test publish_test_geofence_map_pp_b1_v3.test bag_name:='auto_record_2021-06-01-22-49-39_24' target_id:='2365'
popd
