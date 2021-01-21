#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Write lftp script file in the rosbag_dir for uploading rosbag files.
"""
import argparse
import sys
import os
import rospkg
import rospy
from sb_rosbag_sender import SBRosbagSender


def main():
    """Program entry"""
    pkg_dir = rospkg.RosPack().get_path("fail_safe")
    src_dir = os.path.join(pkg_dir, "src")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sb-rosbag-sender-ini",
        default=os.path.join(src_dir, "sb_rosbag_sender_local.ini"))
    parser.add_argument(
        "--rosbag-dir",
        default=os.path.join(os.environ["HOME"], "rosbag_files", "tmp"))
    args = parser.parse_known_args()[0]

    rospy.init_node("SBRosbagSender")

    if not os.path.isdir(args.rosbag_dir):
        os.makedirs(args.rosbag_dir)

    ini = args.sb_rosbag_sender_ini
    if not os.path.isfile(ini):
        rospy.logwarn("Cannot find %s", ini)
        ini = os.path.join(src_dir, "sb_rosbag_sender.ini")
        rospy.logwarn("Use %s instead. Note: it may not fit your car setting", ini)

    rospy.logwarn("Init SBRosbagSender: ini file: %s, rosbag_dir: %s",
                  ini, args.rosbag_dir)

    sender = SBRosbagSender(ini, args.rosbag_dir)
    sender.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
