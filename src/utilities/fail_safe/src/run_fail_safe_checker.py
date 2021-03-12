#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import argparse
import os
import rospkg
from fail_safe_checker import FailSafeChecker


def main():
    pkg_dir = rospkg.RosPack().get_path("fail_safe")
    src_dir = os.path.join(pkg_dir, "src")

    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat-ini", default=os.path.join(src_dir, "heartbeat.ini"))
    parser.add_argument("--mqtt-ini", default=os.path.join(src_dir, "mqtt_b1.ini"))
    parser.add_argument("--mqtt-fqdn", default=None, help="Use the mqtt fqdn given in command line.")
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_known_args()[0]

    checker = FailSafeChecker(args.heartbeat_ini, args.mqtt_ini, args.mqtt_fqdn)
    checker.set_debug_mode(args.debug_mode)

    checker.run()

if __name__ == "__main__":
    main()
