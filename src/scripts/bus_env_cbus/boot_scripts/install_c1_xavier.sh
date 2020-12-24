#!/bin/bash
set -x
set -e

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root"
  exit
fi

cp camera/xavier_rc.local /etc/rc.local
bash /etc/rc.local

