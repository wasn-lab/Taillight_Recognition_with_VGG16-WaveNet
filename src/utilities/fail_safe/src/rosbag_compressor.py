# -*- encoding: utf-8 -*-
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
    cmd = ["gzip", bag_fullpath]
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
        rospy.init_node("RosbagCompressor")
        rospy.logwarn("Init RosbagCompressor, rosbag_dir: %s", rosbag_dir)
        self.rosbag_dir = rosbag_dir

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.compress_bags_if_necessary()
            rate.sleep()

    def compress_bags_if_necessary(self):
        for root, _dirs, files in os.walk(self.rosbag_dir):
            for filename in files:
                if not filename.endswith(".bag"):
                    continue
                bag_fullpath = os.path.join(root, filename)
                ratio = compress_bag(bag_fullpath)
                rospy.loginfo("Compress %s, compression ratio: %f", bag_fullpath, ratio)
