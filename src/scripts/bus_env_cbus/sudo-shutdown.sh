#!/bin/bash
set -x

ssh -t xavier "echo nvidia | sudo -S shutdown now"
ssh -t local "echo itri | sudo -S shutdown now"
ssh -t camera "echo itri | sudo -S shutdown now"
ssh -t throttle "echo itri | sudo -S shutdown now"

car_model=$(rosparam get /car_model)
if [[ "${car_model}" == "C2" || "${car_model}" == "C3" ]]; then
  ssh -t pc2_cmpr "echo itri | sudo -S shutdown now"
fi

echo "All done"
