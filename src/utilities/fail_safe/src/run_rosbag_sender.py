#!/usr/bin/env python
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
    parser.add_argument("--vid", default="itriadv")
    parser.add_argument("--rosbag-sender-ini", default=os.path.join(src_dir, "rosbag_sender.ini"))
    args = parser.parse_known_args()[0]

    cfg = configparser.ConfigParser()
    cfg.read(args.rosbag_sender_ini)

    _rate = cfg["ftp"].getint("upload_rate", 1000000)
    sender = RosbagSender(cfg["ftp"]["fqdn"], cfg["ftp"]["port"],
                           cfg["ftp"]["user_name"],
                           cfg["rosbag"]["backup_dir"],
                           vid=args.vid,
                           upload_rate=_rate)

    sender.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
