#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
python maximize_window.py --window-title <title>
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


def maximize_window(window_title):
    wid = find_window_id(window_title)
    if not wid:
        return
    cmd = ["xdotool", "windowsize", "-sync", wid, "100%", "100%"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    for prog in ["xrandr", "wmctrl", "xdotool"]:
        if not _installed(prog):
            sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-title", "-w", required=True)
    parser.add_argument("--monitor", "-m", required=True)
    args = parser.parse_args()
    maximize_window(args.window_title.decode("utf-8"))

if __name__ == "__main__":
    main()
