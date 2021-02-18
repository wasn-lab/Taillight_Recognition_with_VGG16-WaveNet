#!/bin/sh
dir=`pwd`
#change to init_ar0231_1207_9286trigger.sh directory, wakeup init_ar0231_1207_9286trigger.sh to initiate ar0231 driver
cd /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber
chmod +x ./init_ar0231_1207_9286trigger_v2.sh ./eq_fine_tune_max.sh ./eq_fine_tune_profile.sh
echo nvidia | sudo -S ./init_ar0231_1207_9286trigger_v2.sh
echo "enable camera driver for car mode"
echo nvidia | sudo -S ./eq_fine_tune_max.sh
cd $dir
