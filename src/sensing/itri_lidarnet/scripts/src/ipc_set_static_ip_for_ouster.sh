#!/usr/bin/env bash 

echo "This script will reset static_ip for ousters."
echo ""
sudo apt -y install httpie
echo "Ouster IP List:"
avahi-browse -lr -t _roger._tcp | grep -e 'os1\|addr'
echo "--------------------------------"

left_ip=\"192.168.0.221/24\"
right_ip=\"192.168.0.222/24\"
top_ip=\"192.168.0.223/24\"


read -p "Enter *OLD* IPv4 for Left | " dhcp_left_ip
echo "Will set $dhcp_left_ip to Left."
target_url="http://$dhcp_left_ip/api/v1/system/network/ipv4/override/"
echo $left_ip | http PUT "http://$dhcp_left_ip/api/v1/system/network/ipv4/override"

read -p "Enter *OLD* IPv4 for Right | " dhcp_right_ip
echo "Will set $dhcp_right_ip to Right."
target_url="http://$dhcp_right_ip/api/v1/system/network/ipv4/override/"
echo $right_ip | http PUT "http://$dhcp_right_ip/api/v1/system/network/ipv4/override"

read -p "Enter *OLD* IPv4 for Top, example: xxx.xxx.xxx.xxx | "  dhcp_top_ip
echo "Will set $dhcp_top_ip to TOP."
target_url="http://$dhcp_top_ip/api/v1/system/network/ipv4/override/"
echo $top_ip | http PUT "http://$dhcp_top_ip/api/v1/system/network/ipv4/override"
