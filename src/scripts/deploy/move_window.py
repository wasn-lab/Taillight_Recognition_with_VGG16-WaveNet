#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
python move_window.py --window-title <title> --monitor <monitor>

Use xrandr to find out monitor. Usually monitor is HDMI-0, DP-0 etc.
"""
from __future__ import print_function
import argparse
import sys
import subprocess

def _installed(prog):
    found = False
    try:
        subprocess.check_call(["which", prog])
        print("Successfully find " + prog + ". Keep going.")
        found = True
    except subprocess.CalledProcessError:
        print("Oh! You need to install " + prog)
    return found


def find_window_id(window_title):
    output = subprocess.check_output(["wmctrl", "-l"]).decode("utf-8")
    for line in output.splitlines():
        if window_title not in line:
            continue
        wid = line.split()[0]
        print("Find window id: {}".format(wid))
        return wid
    print("Cannot find window id for {}".format(window_title))
    return ""


def maximize_window(wid):
    cmd = ["xdotool", "windowsize", "-sync", wid, "100%", "100%"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def move_window(wid, monitor):
    xrandr_out = subprocess.check_output(["xrandr"]).decode("utf-8")
    hpos = ""
    for line in xrandr_out.splitlines():
        if ("connected" not in line) or (monitor not in line):
            continue
        fields = line.split()
        for field in fields:
            if field.count("+") != 2:
                continue
            hpos = field.split("+")[-2]
            print("Find horizontal offset {} for monitor {}".format(hpos, monitor))
    if not hpos:
        print("Cannot find horizontal offset for monitor " + monitor)
        return

    cmd = ["wmctrl", "-ir", wid, "-e", "0,"+str(int(hpos)+100)+",100,300,300"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def move_and_max(window_title, monitor):
    wid = find_window_id(window_title)
    if not wid:
        return
    move_window(wid, monitor)
    maximize_window(wid)

def main():
    for prog in ["xrandr", "wmctrl", "xdotool"]:
        if not _installed(prog):
            sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-title", "-w", required=True)
    parser.add_argument("--monitor", "-m", required=True)
    args = parser.parse_args()
    move_and_max(args.window_title.decode("utf-8"), args.monitor)

if __name__ == "__main__":
    main()
