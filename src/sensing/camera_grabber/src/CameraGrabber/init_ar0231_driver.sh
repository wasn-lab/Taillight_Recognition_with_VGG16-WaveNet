#!/bin/sh
dir=`pwd`
#change to init_ar0231_1207_9286trigger.sh directory, wakeup init_ar0231_1207_9286trigger.sh to initiate ar0231 driver
cd $1
chmod +x ./init_ar0231_1207_9286trigger_v2.sh ./eq_fine_tune_max.sh ./eq_fine_tune_profile.sh
echo $2 | sudo -S ./init_ar0231_1207_9286trigger_v2.sh
if [ "$3" = "true" ]
then
  echo "enable car camera driver"
  echo nvidia | sudo -S ./eq_fine_tune_max.sh
else
  echo "enable laboratory camera driver"	
  echo nvidia | sudo -S ./eq_fine_tune_profile.sh
fi
cd $dir
