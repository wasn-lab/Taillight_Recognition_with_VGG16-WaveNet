#!/bin/bash
# this bash will install GLFW3 which required from ouster driver.

set -x
set -e

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root"
  exit
fi

echo "GLFW3 will be installed......"
sleep 1
cd ~
git clone https://github.com/glfw/glfw.git
cd ~/glfw
cmake -G "Unix Makefiles"
apt-get install cmake xorg-dev libglu1-mesa-dev
make
make install
echo "Done"

