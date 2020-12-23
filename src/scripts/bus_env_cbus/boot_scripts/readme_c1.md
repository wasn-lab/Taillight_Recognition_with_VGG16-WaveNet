### Special setting for car C1
Run the following commands for setting up the boot sequence:

- For lidar IPC
  - set 192.168.1.3 for the iface with router
  - set 192.168.3.1 for the iface with camera
  - cp src/scripts/bus_env_cbus/boot_scripts/rosmasterd.sh /usr/local/bin/
  - cp src/scripts/bus_env_cbus/boot_scripts/lidar_rc.local /etc/rc.local

- For camera IPC
  - set 192.168.3.10 for the iface with lidar
  - 192.168.2.1 for the iface with xavier
- For xavier
  - set 192.168.1.6 for the iface with router
  - set 192.168.2.10 for the iface with camera
- For localization
  - set 192.168.1.5 for the iface with router
