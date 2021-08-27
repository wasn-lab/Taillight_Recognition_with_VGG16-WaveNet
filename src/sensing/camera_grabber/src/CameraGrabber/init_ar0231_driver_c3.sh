#!/bin/sh
dir=`pwd`
#change to init_ar0231_20210331v3_2.sh directory, wakeup init_ar0231_20210331v3_2.sh to initiate ar0231 driver
cd $1
chmod +x ./init_ar0231_20210331v3_2.sh ./eq_fine_tune_profile_20210511_13+8db.sh ./eq_fine_tune_profile.sh ./exposure-setup-day-or-night.sh
echo $2 | sudo -S ./init_ar0231_20210331v3_2.sh
if [ "$3" = "true" ]
then
  echo "enable car camera driver for car model c3"
  echo $2 | sudo -S ./eq_fine_tune_profile_20210511_13+8db.sh
else
  echo "enable laboratory camera driver"	
  echo $2 | sudo -S ./eq_fine_tune_profile.sh
fi
cd $dir
