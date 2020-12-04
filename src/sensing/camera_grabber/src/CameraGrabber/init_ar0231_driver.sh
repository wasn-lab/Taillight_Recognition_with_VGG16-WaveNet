#!/bin/sh
dir=`pwd`
#change to init_ar0231 directory, wakeup init_ar0231 to initiate ar0231 driver
cd $1
chmod +x ./init_ar0231.sh ./eq_fine_tune_max.sh
echo $2 | sudo -S ./init_ar0231.sh
echo $2 | sudo -S ./eq_fine_tune_max.sh
#echo $2 | sudo -S ./eq_fine_tune_profile.sh
cd $dir
