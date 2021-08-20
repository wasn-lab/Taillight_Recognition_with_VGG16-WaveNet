#! /bin/bash
readonly car_model=$(rosparam get /car_model)
if [[ "${car_model}" == "C2" ]]; then
  source /home/throttle/itriadv/devel/setup.bash
elif [[ "${car_model}" == "C3" ]]; then
  source /home/throttle/itriadv/devel/setup.bash
else
  source /home/itri/itriadv/devel/setup.bash
fi
#source /home/itri/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311
export ROS_IP=192.168.1.7

roslaunch daq_io daq_io.launch
