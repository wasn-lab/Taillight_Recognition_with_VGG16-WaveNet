#!/bin/bash
set -x
set -e

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1" ]]; then
  echo "This script is for B1 only."
  exit 0
fi

source devel/setup.bash
roslaunch car_model drivenet_hz.test
