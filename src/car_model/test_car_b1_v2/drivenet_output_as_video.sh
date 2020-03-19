#!/bin/bash
set -x
set -e

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi

source devel/setup.bash
export output_dir=/var/www/html/artifacts/$(date "+%Y%m%d")
mkdir -p $output_dir

roslaunch car_model drivenet_output_as_video.test
