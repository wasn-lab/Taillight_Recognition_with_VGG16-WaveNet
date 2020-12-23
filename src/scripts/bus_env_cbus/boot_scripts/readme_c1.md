### Special setting for car C1
Run the following commands for setting up the boot sequence:

- For lidar IPC (run as root)
  - cp src/scripts/bus_env_cbus/boot_scripts/systemd/system/rosmasterd.service /etc/systemd/system/
  - cp src/scripts/bus_env_cbus/boot_scripts/rosmasterd.sh /usr/local/bin/
  - echo "route add -net 192.168.2.0 netmask 255.255.255.0 gw 192.168.3.10" >> /etc/rc.local
  - chmod +x /etc/rc.local
