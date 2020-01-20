#!/bin/bash

sudo apt-get -y purge dnsmasq
sudo apt-get -y install dnsmasq
sudo bash -c "rm /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "touch /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set the interface on which dnsmasq operates.' >> /etc/dnsmasq.d/dhcp.conf"
	
sudo bash -c "echo 'interface=bridge0' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# To disable dnsmasq DNS server functionality.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'port=0' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# To enable dnsmasq DHCP server functionality.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-range=192.168.0.200,192.168.0.250,255.255.255.0,12h' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set static IPs of other PC' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# dhcp_client_mac_address,dhcp_client_name,dhcp_client_ip,dhcp_client_lease' >> /etc/dnsmasq.d/dhcp.conf"

sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:05:f2,os1-991925000304,192.168.0.220,infinite' >> /etc/dnsmasq.d/dhcp.conf" #220
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:11:ba,os1-991941000026,192.168.0.221,infinite' >> /etc/dnsmasq.d/dhcp.conf" #221
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:ad,os1-991940000164,192.168.0.222,infinite' >> /etc/dnsmasq.d/dhcp.conf" #222	
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:11:26,os1-991940000799,192.168.0.223,infinite' >> /etc/dnsmasq.d/dhcp.conf" #223
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:11:90,os1-991941000030,192.168.0.224,infinite' >> /etc/dnsmasq.d/dhcp.conf" #224	
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:b4,os1-991939001245,192.168.0.225,infinite' >> /etc/dnsmasq.d/dhcp.conf" #225	
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:f5,os1-991941001040,192.168.0.226,infinite' >> /etc/dnsmasq.d/dhcp.conf" #226	
	
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set gateway as Router. Following two lines are identical.' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-option=3,192.168.0.1' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
#sudo bash -c "echo '# Set DNS server as Router.' >> /etc/dnsmasq.d/dhcp.conf"
#sudo bash -c "echo 'dhcp-option=6,192.168.0.1' >> /etc/dnsmasq.d/dhcp.conf"
#sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"


sudo sed -i "s/127.0.0.1/192.168.11.1/g" /etc/resolv.conf #router ip
sudo bash -c "echo 'nameserver 8.8.8.8' >> /etc/resolv.conf"
sudo update-rc.d dnsmasq defaults

grep -q "After=network-online.target" /lib/systemd/system/dnsmasq.service
if [ $? -eq 1 ]
then
  sudo sed -i -e "/Requires=network.target/a After=network-online.target\nWants=network-online.target" /lib/systemd/system/dnsmasq.service
fi

sudo systemctl enable dnsmasq.service
sudo systemctl start dnsmasq.service
#sudo restart network-manager
#sudo systemctl stop dnsmasq.service
#sudo systemctl status dnsmasq.service
