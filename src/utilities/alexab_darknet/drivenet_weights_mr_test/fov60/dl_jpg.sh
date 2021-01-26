#!/bin/bash
set -x
set -e
readonly cur_dir=$(readlink -e $(dirname $0))
pushd $cur_dir
for json in `ls *.json`; do
  jpg=$(basename $json .json).jpg
  if [[ ! -f $jpg ]]; then
    wget http://nas.itriadv.co:8888/git_data/B1/drivenet_weights_mr_test/fov60/$jpg
  fi
done
popd
