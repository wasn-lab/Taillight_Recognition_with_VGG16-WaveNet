# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Upload bag files to MOT

Work in progress
"""
import argparse
import pprint

import rosbag

def _analyze(bag_filename):
    bag = rosbag.Bag(bag_filename)
    for item in bag.read_messages("/vehicle/report/itri/sensor_status"):
        _msg = item.message
        _timestamp = item.timestamp
        _topic = item.topic
        print(_timestamp)

    bag.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", required=True)
    args = parser.parse_args()
    _analyze(args.rosbag)

if __name__ == "__main__":
    main()
