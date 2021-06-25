#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir/build
set +x
echo "source $repo_dir/devel/setup.bash"
source $repo_dir/devel/setup.bash
set -x
export LD_PRELOAD=/usr/local/lib/libopencv_core.so

make auto_record_2020-03-10-10-48-39_41_image_raw.bag
rostest image_compressor publish_test_cmpr.test

make auto_record_2020-12-28-16-55-14_29_jpg.bag
rostest image_compressor publish_test_decmpr.test

popd
