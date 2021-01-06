#! /bin/bash
echo start setting network........

#gnome-terminal -e  'echo itri | sudo -S route add default gw 192.168.3.1'
#sleep 3
gnome-terminal  -e 'ssh -t lidar "source .bashrc;echo itri | sudo -S route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10"'
gnome-terminal  -e 'ssh -t lidar "source .bashrc;echo itri | sudo -S bash ip_forwarding.sh"'
sleep 3
gnome-terminal  -e "bash -c 'echo itri | sudo -S bash ip_forwarding.sh'"
#gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S ifconfig eth0:400 192.168.1.222"'
#sleep 3
#gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S ifconfig enp4s0 192.168.2.10"'
#sleep 3
gnome-terminal  -e 'ssh -t ta "source .bashrc;echo nvidia | sudo -S route add -net 192.168.3.0 netmask 255.255.255.0 gw 192.168.1.3"'
