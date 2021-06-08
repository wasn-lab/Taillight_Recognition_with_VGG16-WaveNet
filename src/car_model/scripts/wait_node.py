#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import argparse
import sys
import os
import time
from rosnode import get_node_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", required=True)
    args = parser.parse_args()

    master = os.environ.get("ROS_MASTER_URI", "")
    print("ROS_MASTER_URI: " + master)
    while args.node_name not in get_node_names():
        print("Wait node {} to be online.".format(args.node_name))
        time.sleep(1)
    print("Node {} is online.".format(args.node_name))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
