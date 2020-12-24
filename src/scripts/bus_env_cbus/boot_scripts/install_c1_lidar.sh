#!/bin/bash
set -x
set -e

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root"
  exit
fi

cp lidar/rosmasterd.sh /usr/local/bin/
cp lidar/systemd/system/rosmasterd.service /etc/systemd/system
systemctl enable rosmasterd

cp lidar/lidar_ip_forwarding.sh /usr/local/bin/ip_forwarding.sh
cp lidar/mps.sh /usr/local/bin/
cp lidar/lidar_rc.local /etc/rc.local
bash /etc/rc.local

