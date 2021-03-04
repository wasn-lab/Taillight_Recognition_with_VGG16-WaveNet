#!/bin/bash
set -x

script_dir=$(dirname $0)
repo_dir=${script_dir}/../../../..

pushd $repo_dir
catkin_make -DCMAKE_BUILD_TYPE=Release -DCAR_MODEL=C1
popd
