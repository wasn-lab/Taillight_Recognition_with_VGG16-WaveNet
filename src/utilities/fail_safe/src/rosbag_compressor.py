# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Send backup rosbag files to backend.
"""
from __future__ import print_function
import time
import os
import subprocess
import rospy


def should_compress_bag(root, filename):
    if not filename.endswith(".bag"):
        return False
    if "backup" not in root:
        return False
    return True


def decompress_bag(bag_fullpath):
    if not os.path.isfile(bag_fullpath):
        return
    cmd = ["gzip", "-d", bag_fullpath]
    subprocess.check_call(cmd)


def compress_bag(bag_fullpath):
    if not os.path.isfile(bag_fullpath):
        return -1
    if not bag_fullpath.endswith(".bag"):
        return -1
    # 19 is least favorable to the process
    cmd = ["nice", "-n", "19", "gzip", bag_fullpath]
    org_size = os.path.getsize(bag_fullpath)
    cmpr_file = bag_fullpath + ".gz"
    ratio = -1
    try:
        subprocess.check_call(cmd)
        cmpr_size = os.path.getsize(cmpr_file)
        ratio = float(cmpr_size) / org_size
    except subprocess.CalledProcessError:
        if os.path.isfile(bag_fullpath) and os.path.isfile(cmpr_file):
            os.unlink(cmpr_file)
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
                if should_compress_bag(root, filename):
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
