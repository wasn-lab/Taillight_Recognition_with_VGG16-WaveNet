#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Aggregate cpu/gpu loads and publish it
"""
import argparse
from load_collector import LoadCollector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqtt-fqdn", default="192.168.1.3")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    args = parser.parse_known_args()[0]
    col = LoadCollector(args.mqtt_fqdn, args.mqtt_port)
    col.run()
