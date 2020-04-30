sudo iptables -A FORWARD -o enp1s0f0 -i enp1s0f1 -s 192.168.3.1/24 -m conntrack --ctstate NEW -j ACCEPT
sudo iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -t nat -F POSTROUTING
sudo iptables -t nat -A POSTROUTING -o enp1s0f0 -j MASQUERADE

