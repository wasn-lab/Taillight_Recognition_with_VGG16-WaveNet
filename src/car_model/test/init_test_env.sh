#!/bin/bash
set -x
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" == "B1_V2" ]]; then
  bash src/car_model/test_car_b1_v2/init_test_env.sh
elif [[ "${CAR_MODEL}" == "B1_V3" ]]; then
  bash src/car_model/test_car_b1_v3/init_test_env.sh
fi
