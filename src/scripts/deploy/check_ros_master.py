#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import argparse
import sys
import os
import time
import rosgraph


def is_master_alive():
    ret = False
    if rosgraph.is_master_online():
        print("ROS MASTER is Online")
        ret = True
    else:
        print("ROS MASTER is Offline")
        ret = False
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-until-alive", action="store_true")
    args = parser.parse_args()

    master = os.environ.get("ROS_MASTER_URI", "")
    print("ROS_MASTER_URI: " + master)
    alive = is_master_alive()
    while (not alive) and args.wait_until_alive:
        print("Wait 1 second and check master again.")
        time.sleep(1)
        alive = is_master_alive()
    return 0 if alive else 1


if __name__ == "__main__":
    sys.exit(main())
