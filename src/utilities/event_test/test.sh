#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)

pushd $repo_dir
set +x
source devel/setup.bash
set -x

# positive cases
rostest event_test publish_test_geofence_map_pp_b1_v3.test is_cut:=true bag_name:='auto_record_2021-06-01-22-49-39_24' target_id:='2365' start_time:=10 duration:=2.5

# negative cases
rostest event_test publish_test_geofence_map_pp_b1_v3.test is_cut:=false bag_name:='auto_record_2021-06-01-22-49-39_24' target_id:='94db' start_time:=6.5 duration:=3
popd
