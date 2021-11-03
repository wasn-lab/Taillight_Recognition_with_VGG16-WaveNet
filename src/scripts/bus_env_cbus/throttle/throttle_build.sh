#!/bin/bash
set -x

script_dir=$(dirname $0)
repo_dir=${script_dir}/../../../..

car_model=$(rosparam get /car_model)
if [[ -z "${car_model}" ]]; then
  car_model=C1
fi


pushd ${repo_dir}
catkin_make -DCAR_MODEL=${car_model} -DCATKIN_WHITELIST_PACKAGES="msgs;plc_fatek;daq_io"
popd
