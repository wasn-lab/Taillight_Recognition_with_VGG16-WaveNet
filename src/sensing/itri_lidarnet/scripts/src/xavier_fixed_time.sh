sudo /etc/init.d/networking restart
sudo killall ptp4l
sudo killall phc2sys
sudo service ntp stop
sudo timedatectl set-ntp false
sudo ntpdate pool.ntp.org