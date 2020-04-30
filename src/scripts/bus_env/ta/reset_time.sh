#sudo dhclient enp4s0
#sudo /etc/init.d/networking restart
echo nvidia | sudo -S killall ptp4l
echo nvidia | sudo -S killall phc2sys
echo nvidia | sudo -S service ntp stop
echo nvidia | sudo -S timedatectl set-ntp false
echo nvidia | sudo -s ntpdate pool.ntp.org
