#!/bin/bash
set -x

script_dir=$(dirname $(readlink -e $0))
repo_dir=$(readlink -e ${script_dir}/../../../..)

pushd $repo_dir
catkin_make -DCMAKE_BUILD_TYPE=Release -DCAR_MODEL=C2
popd
