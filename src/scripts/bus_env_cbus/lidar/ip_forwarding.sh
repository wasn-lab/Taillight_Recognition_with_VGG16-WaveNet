iptables -A FORWARD -o enp8s0 -i enp1s0f1 -s 192.168.3.10/24 -m conntrack --ctstate NEW -j ACCEPT
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -t nat -F POSTROUTING
iptables -t nat -A POSTROUTING -o enp8s0 -j MASQUERADE

iptables -A FORWARD -o enp8s0 -i enp0s31f6 -s 192.168.4.10/24 -m conntrack --ctstate NEW -j ACCEPT
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -t nat -F POSTROUTING
iptables -t nat -A POSTROUTING -o enp8s0 -j MASQUERADE
