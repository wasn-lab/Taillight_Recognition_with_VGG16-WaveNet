sudo iptables -A FORWARD -o enp1s0f1 -i enp1s0f0 -s 192.168.2.1/24 -m conntrack --ctstate NEW -j ACCEPT
sudo iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -t nat -F POSTROUTING
sudo iptables -t nat -A POSTROUTING -o enp1s0f1 -j MASQUERADE

