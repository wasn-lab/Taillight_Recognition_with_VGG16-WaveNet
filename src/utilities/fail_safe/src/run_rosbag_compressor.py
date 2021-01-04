#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Compress *.bag into *.bag.gz
"""
import argparse
import sys
import os
import rospkg
import rospy
from rosbag_compressor import RosbagCompressor


def main():
    """Program entry"""
    pkg_dir = rospkg.RosPack().get_path("fail_safe")
    src_dir = os.path.join(pkg_dir, "src")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosbag-dir",
        default=os.path.join(os.environ["HOME"], "rosbag_files", "tmp"))
    args = parser.parse_known_args()[0]

    rospy.init_node("RosbagCompressor")
    rospy.logwarn("Init RosbagCompressor, rosbag_dir: %s", args.rosbag_dir)

    if not os.path.isdir(args.rosbag_dir):
        os.makedirs(args.rosbag_dir)

    compressor = RosbagCompressor(args.rosbag_dir)
    compressor.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
