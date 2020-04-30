sudo iptables -A FORWARD -o enp8s0 -i enp1s0f1 -s 192.168.3.10/24 -m conntrack --ctstate NEW -j ACCEPT
sudo iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -t nat -F POSTROUTING
sudo iptables -t nat -A POSTROUTING -o enp8s0 -j MASQUERADE
