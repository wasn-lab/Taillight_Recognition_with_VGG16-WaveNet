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
sudo bash -c "echo 'dhcp-range=192.168.0.100,192.168.0.230,255.255.255.0,12h' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# Set static IPs of other PC' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo '# dhcp_client_mac_address,dhcp_client_name,dhcp_client_ip,dhcp_client_lease' >> /etc/dnsmasq.d/dhcp.conf"

#--- Ouster List
# b1 @ 20210511
sudo bash -c "echo '#---- b1 @ 20210511' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:1e:9a,os1-122035000204,192.168.0.226,infinite' >> /etc/dnsmasq.d/dhcp.conf" 
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:f5,os1-991941001040,192.168.0.230,infinite' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:b4,os1-991939001245,192.168.0.225,infinite' >> /etc/dnsmasq.d/dhcp.conf" 
# c1 
sudo bash -c "echo '#---- c1 ' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:15:af,os1-122016000062,192.168.0.228,infinite' >> /etc/dnsmasq.d/dhcp.conf" # Left
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:1e:ed,os1-122035000199,192.168.0.227,infinite' >> /etc/dnsmasq.d/dhcp.conf" # Right
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:1e:52,os1-122032000008,192.168.0.229,infinite' >> /etc/dnsmasq.d/dhcp.conf" # Top
# lab @ 20210511
sudo bash -c "echo '#---- lab @ 20210511' >> /etc/dnsmasq.d/dhcp.conf"
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:05:4f,os1-122033000240,192.168.0.104,infinite' >> /etc/dnsmasq.d/dhcp.conf" 
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:1f:dd,os1-122043000296,192.168.0.122,infinite' >> /etc/dnsmasq.d/dhcp.conf" 
sudo bash -c "echo 'dhcp-host=bc:0f:a7:00:0d:57,os1-992003000215,192.168.0.177,infinite' >> /etc/dnsmasq.d/dhcp.conf" 
#--- End of Ouster List


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