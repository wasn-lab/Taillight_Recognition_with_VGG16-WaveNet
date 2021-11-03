#!/bin/bash
set -x

script_dir=$(dirname $(readlink -e $0))
repo_dir=$(readlink -e ${script_dir}/../../../..)

car_model=$(rosparam get /car_model)
if [[ -z "${car_model}" ]]; then
  car_model=C1
fi

pushd $repo_dir
catkin_make -DCAR_MODEL=${car_model}
popd
