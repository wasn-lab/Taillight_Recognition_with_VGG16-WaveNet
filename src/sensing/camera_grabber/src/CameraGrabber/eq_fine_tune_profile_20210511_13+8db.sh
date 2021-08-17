#!/bin/sh

# Disable Line Activity Detector
echo "Disable Line Activity Detector"
sudo i2cset -f -y 2 0x48 0x11 0x80
sleep 0.01
sudo i2cset -f -y 7 0x48 0x11 0x80
sleep 0.01

# Enable Line Activity Detector after found optimal value
# echo "Disable Line Activity Detector"
# sudo i2cset -f -y 2 0x48 0x11 0x20
# sleep 0.01
# sudo i2cset -f -y 7 0x48 0x11 0x20
# sleep 0.01

echo "EQ 13dB"
sudo i2cset -f -y 2 0x48 0x32 0xbb
sleep 0.01
sudo i2cset -f -y 2 0x48 0x33 0xbb
sleep 0.01
sudo i2cset -f -y 7 0x48 0x32 0xbb
sleep 0.01
sudo i2cset -f -y 7 0x48 0x33 0xbb
sleep 0.01

echo "Pre-Emp 8dB"
sudo i2cset -f -y 2 0x65 0x06 0xad
sleep 0.01
sudo i2cset -f -y 7 0x65 0x06 0xad
sleep 5

#echo "14.0dB Pre-Emp"
#sudo i2cset -f -y 2 0x65 0x06 0xaf
#sudo i2cset -f -y 7 0x65 0x06 0xaf
