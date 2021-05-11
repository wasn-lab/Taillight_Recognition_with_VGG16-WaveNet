#!/bin/bash
set -x
readonly repo_dir=$(git rev-parse --show-toplevel)
export ASAN_OPTIONS=protect_shadow_gap=0,log_path=/tmp/pedcross_asan.log
export LSAN_OPTIONS=suppressions=${repo_dir}/src/scripts/ci/lsan.suppress
set +x
source ${repo_dir}/devel/setup.bash
set -x
rostest car_model publish_test_pedcross_b1_v3_asan.test > /dev/null  # no pedcross UI
ret=0
# log file format is something like /tmp/pedcross_asan.log.12484
for logf in `ls /tmp/pedcross_asan.log*`; do
  cat $logf
  grep "ERROR:" $logf
  if [[ "$?" == "0" ]]; then
    ret=1
  fi
  rm $logf
done

for openpose_file in `ls /tmp/OpenPose*`; do
  rm $openpose_file
done

exit $ret
