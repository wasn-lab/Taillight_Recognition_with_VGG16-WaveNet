#!/usr/bin/env python
"""
Generate iptables that allows lidar ipc to relay packets for camera ipc.
"""
import os
import subprocess
import re

def _get_interfaces():
    ifnames = os.listdir("/sys/class/net/")
    ip_re = re.compile(".*inet\s+(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
    interfaces = []

    for ifname in ifnames:
        ip = ""
        output = subprocess.check_output(["ifconfig", ifname])
        for line in output.splitlines():
            match = ip_re.match(line)
            if match:
                ip = match.expand(r"\g<ip>")
        interfaces.append({"interface": ifname, "ip": ip, "speed": "-1"})
    for doc in interfaces:
        ifname = doc["interface"]
        if ifname == "lo" or ifname == "br0":
            continue
        tmp = "/sys/class/net/{}/speed".format(doc["interface"])
        try:
            output = subprocess.check_output(["cat", tmp]).decode("utf-8")
            doc["speed"] = output.splitlines()[0]
        except subprocess.CalledProcessError:
            pass
    return interfaces


def main():
    interfaces = _get_interfaces()
    print(interfaces)

if __name__ == "__main__":
    main()
