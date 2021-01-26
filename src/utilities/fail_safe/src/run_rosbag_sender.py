#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import argparse
import sys
import os
import configparser
import rospkg
from rosbag_sender import RosbagSender


def main():
    pkg_dir = rospkg.RosPack().get_path("fail_safe")
    src_dir = os.path.join(pkg_dir, "src")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag-sender-ini", default=os.path.join(src_dir, "rosbag_sender.ini"))
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_known_args()[0]

    cfg = configparser.ConfigParser()
    cfg.read(args.rosbag_sender_ini)

    sender = RosbagSender(cfg["ftp"]["fqdn"], cfg["ftp"]["port"],
                          cfg["ftp"]["user_name"],
                          cfg["ftp"]["password"],
                          cfg["rosbag"]["backup_dir"],
                          cfg["ftp"]["upload_rate"])
    sender.set_debug_mode(args.debug_mode)
    sender.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
