#!/bin/bash
sudo apt install -y bridge-utils
sudo ifdown -a

# enter ifconfig info
ls /sys/class/net/
echo ""
read -p "Enter interface_names (split by space): " interface_names
IFS=' ' read -a interface_names_array <<< "$interface_names"  

for index in ${!interface_names_array[@]}
do  
    echo "$index : ${interface_names_array[index]}"  
    line+="${interface_names_array[index]} "
done 
echo "interface counts: ${#interface_names_array[@]}"  
#echo "bridge_ports $line"


# create br0 for ouster
sudo bash -c "echo '' >> /etc/network/interfaces"
sudo bash -c "echo '# create br0 for ouster' >> /etc/network/interfaces"
sudo bash -c "echo 'auto br0' >> /etc/network/interfaces"
sudo bash -c "echo 'iface br0 inet static' >> /etc/network/interfaces"
sudo bash -c "echo 'address 192.168.0.1' >> /etc/network/interfaces"
sudo bash -c "echo 'netmask 255.255.255.0' >> /etc/network/interfaces"
sudo bash -c "echo 'network 192.168.0.1' >> /etc/network/interfaces"
sudo bash -c "echo 'broadcast 255.255.255.255' >> /etc/network/interfaces"
sudo bash -c "echo 'gateway 192.168.0.2' >> /etc/network/interfaces"
sudo bash -c "echo 'dns-nameservers 8.8.8.8 8.8.4.4' >> /etc/network/interfaces"
sudo bash -c "echo 'dns-search localdomain.local' >> /etc/network/interfaces"
sudo bash -c "echo 'bridge_ports $line' >> /etc/network/interfaces"


sudo ifup -a
sudo ifup --ignore-errors br0
