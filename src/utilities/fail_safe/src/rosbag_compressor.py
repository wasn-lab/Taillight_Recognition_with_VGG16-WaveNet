# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Send backup rosbag files to backend.
"""
from __future__ import print_function
import time
import datetime
import os
import io
import subprocess
import rospy


def decompress_bag(bag_fullpath):
    cmd = ["gzip", "-d" , bag_fullpath]
    subprocess.check_call(cmd)

def compress_bag(bag_fullpath):
    if not bag_fullpath.endswith(".bag"):
        return -1
    # 19 is least favorable to the process
    cmd = ["nice", "-n", "19", "gzip", bag_fullpath]
    org_size = os.path.getsize(bag_fullpath)
    subprocess.check_call(cmd)
    cmpr_file = bag_fullpath + ".gz"
    cmpr_size = os.path.getsize(cmpr_file)
    ratio = float(cmpr_size) / org_size
    return ratio


class RosbagCompressor(object):
    def __init__(self, rosbag_dir):
        """
        Compres *.bag in *.bag.gz for transmission.
        """
        self.rosbag_dir = rosbag_dir

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.compress_bags_if_necessary()
            rate.sleep()

    def compress_bags_if_necessary(self):
        bags = []
        for root, _dirs, files in os.walk(self.rosbag_dir):
            for filename in files:
                if not filename.endswith(".bag"):
                    continue
                bags.append(os.path.join(root, filename))
        bags.sort()
        for bag in bags:
            start_time = time.time()
            rospy.loginfo("Start compressing %s", bag)
            ratio = compress_bag(bag)
            elapsed_time = time.time() - start_time
            gz_size = os.path.getsize(bag + ".gz") / (1024 * 1024)
            rospy.loginfo(("Done compressing %s, ratio: %.2f, file size: %d MB, "
                           "elapsed_time: %f seconds"),
                           bag, ratio, gz_size, elapsed_time)
