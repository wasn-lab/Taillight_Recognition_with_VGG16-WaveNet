#!/bin/sh
dir=`pwd`
#change to init_ar0231 directory, wakeup init_ar0231 to initiate ar0231 driver
cd $1
chmod +x ./init_ar0231
echo $2 | sudo -S ./init_ar0231
cd $dir
