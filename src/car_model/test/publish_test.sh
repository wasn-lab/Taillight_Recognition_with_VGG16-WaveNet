#!/bin/bash
set -x
source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" == "B1_V3" ]]; then
  bash src/car_model/test_car_b1_v3/publish_test.sh
else
  echo "This is for B1_V3 only."
fi
