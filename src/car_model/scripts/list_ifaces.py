#!/usr/bin/env python
"""
Generate iptables that allows lidar ipc to relay packets for camera ipc.
"""
import os
import subprocess
import re

def _get_interfaces():
    output = subprocess.check_output(["ifconfig", "-a"]).decode("utf-8")
    interface_re = re.compile("(?P<ifname>[\w\d]+)\:\s+flags=\d+")
    ip_re = re.compile(".*inet\s+(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
    interfaces = []

    ifname = ""
    ip = ""
    for line in output.splitlines():
        match = interface_re.match(line)
        if match:
            ifname = match.expand(r"\g<ifname>")
            continue
        match = ip_re.match(line)
        if match:
            ip = match.expand(r"\g<ip>")
        if ifname and ip:
            interfaces.append({"interface": ifname, "ip": ip, "speed": -1})
            ifname = ""
            ip = ""
    for doc in interfaces:
        tmp = "/sys/class/net/{}/speed".format(doc["interface"])
        if doc["interface"] == "lo":
            continue
        try:
            output = subprocess.check_output(["cat", tmp]).decode("utf-8")
            doc["speed"] = output.splitlines()[0]
        except subprocess.CalledProcessError:
            pass
    return interfaces


def gen_iptable_rules():
    interfaces = _get_interfaces()
    print(interfaces)

if __name__ == "__main__":
    gen_iptable_rules()
