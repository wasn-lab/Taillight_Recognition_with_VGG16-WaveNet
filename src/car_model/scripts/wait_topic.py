#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import argparse
import sys
import os
import time
import random
import rospy
import rosgraph

from std_msgs.msg import Bool, Empty, Float64, Int32
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from msgs.msg import (DetectedObjectArray, VehInfo, BackendInfo,
                      CompressedPointCloud, CompressedPointCloud2)
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Log

__MSG_TO_CLASS = {
    "geometry_msgs/PoseStamped": PoseStamped,
    "std_msgs/Bool": Bool,
    "std_msgs/Empty": Empty,
    "std_msgs/Float64": Float64,
    "std_msgs/Int32": Int32,
    "sensor_msgs/CompressedImage": CompressedImage,
    "sensor_msgs/Image": Image,
    "sensor_msgs/PointCloud2": PointCloud2,
    "rosgraph_msgs/Log": Log,
    "msgs/CompressedPointCloud": CompressedPointCloud,
    "msgs/CompressedPointCloud2": CompressedPointCloud2,
    "msgs/BackendInfo": BackendInfo,
    "msgs/DetectedObjectArray": DetectedObjectArray,
    "msgs/VehInfo": VehInfo}


def __check_topic_registration(topic_name):
    master = rosgraph.Master("/rosnode")
    found = False
    ret = ""
    print("Check {} registration".format(topic_name))
    while not found:
        for name_type in master.getPublishedTopics('/'):
            if topic_name == name_type[0]:
                found = True
                ret = name_type[1]
        time.sleep(1)
    print("Topic {} is of type {}".format(topic_name, ret))
    return ret


def __check_topic_publication(topic_name, msg_type):
    print("Check a real publication of {}".format(topic_name))
    msg = None

    while msg is None:
        try:
            msg = rospy.wait_for_message(topic_name, __MSG_TO_CLASS[msg_type], timeout=1)
        except rospy.exceptions.ROSException:
            msg = None
    print("Got a message of {}".format(topic_name))


def wait_topic(topic_name):
    node_name = "msg_checker_{}".format(random.randint(1, 10000000))
    rospy.init_node(node_name, anonymous=True)

    msg_type = __check_topic_registration(topic_name)
    __check_topic_publication(topic_name, msg_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-name", "-t", required=True)
    args = parser.parse_args()

    master = os.environ.get("ROS_MASTER_URI", "")
    print("ROS_MASTER_URI: " + master)
    wait_topic(args.topic_name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
