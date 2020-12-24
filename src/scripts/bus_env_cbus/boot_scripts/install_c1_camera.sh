#!/bin/bash
set -x
set -e

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root"
  exit
fi

cp camera/camera_ip_forwarding.sh /usr/local/bin/ip_forwarding.sh
cp camera/camera_rc.local /etc/rc.local
bash /etc/rc.local

