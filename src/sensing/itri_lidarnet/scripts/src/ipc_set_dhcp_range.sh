#!/bin/bash

sudo apt-get -y purge dnsmasq
sudo apt-get -y install dnsmasq
sudo bash -c "rm /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "touch /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set the interface on which dnsmasq operates.' >> /etc/dnsmasq.d/dhcp.conf"
	
sudo bash -c "echo 'interface=br0' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# To disable dnsmasq DNS server functionality.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'port=0' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# To enable dnsmasq DHCP server functionality.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-range=192.168.0.200,192.168.0.230,255.255.255.0,1h' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"

sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set gateway as Router. Following two lines are identical.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-option=3,192.168.0.1' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"


sudo sed -i "s/127.0.0.1/192.168.11.1/g" /etc/resolv.conf #router ip
sudo bash -c "echo 'nameserver 8.8.8.8' >> /etc/resolv.conf"
sudo update-rc.d dnsmasq defaults

#grep -q "After=network-online.target" /lib/systemd/system/dnsmasq.service
#if [ $? -eq 1 ]
#then
#  sudo sed -i -e "/Requires=network.target/a After=network-online.target\nWants=network-online.target" /lib/systemd/system/dnsmasq.service
#fi

sudo systemctl enable dnsmasq.service
sudo systemctl start dnsmasq.service