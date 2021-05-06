#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import sys
import os
import rosgraph


def main():
    ret = 0
    master = os.environ.get("ROS_MASTER_URI", "")
    print("ROS_MASTER_URI: " + master)
    if rosgraph.is_master_online():
        print("ROS MASTER is Online")
    else:
        print("ROS MASTER is Offline")
        ret = 1
    return ret


if __name__ == "__main__":
    sys.exit(main())
