#!/bin/bash
#use disks to format device /dev/vblkdev1
#!!!!!! follow the step below or mv the script to / !!!!!!

UUID=($(sudo blkid -s UUID /dev/vblkdev1))
UUID=${UUID[1]}
UUID_length=$(expr length $UUID - 1)
UUID=$(echo $UUID | cut -c7-$UUID_length)
#echo $UUID

sudo bash -c "echo 'UUID=$UUID /media/home ext4 defaults 0 0' >> /etc/fstab"
sudo mkdir /media/home
sudo mount -a
sudo rsync -aXS /home/. /media/home/.
sleep(10)
cd /
sudo mv /home /home_back
sudo mkdir /home
sudo umount /dev/vblkdev1
sudo sed -i "2d" /etc/fstab
sudo bash -c "echo 'UUID=$UUID /home ext4 defaults 0 0' >> /etc/fstab"
sudo mount -a
#sudo rm -rf /home_back
sudo rm -R /media/home/
