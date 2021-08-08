#!/bin/bash
dir=`pwd`
cd $1
secsSinceMidnight=$(( $(date +%s) - $(date -d '00:00:00' +%s) ))
#echo $secsSinceMidnight
# 18:00 = 18x60x60 = 64800
# 06:30 = 6x60x60 =  21600
if [[ $secsSinceMidnight -lt 21600 || $secsSinceMidnight -gt 64800 ]]; then
	echo "night: camera use manu exposure 8.33ms and gain 0xFF00(max)"
        echo nvidia | sudo -S ./manualExposureOnCam0~7-setup.sh 0x04 0x42 0xFF 0x00
        echo "night: finsh set manu exposure and gain"
else
        echo "day: camera use auto exposure"
        #echo nvidia | sudo -S ./autoExposureOn8Cam.sh	
fi
cd $dir
