#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import argparse
import sys
import rospy
from msgs.msg import Flag_Info


def gen_flag02(fps, self_driving_mode):
    node_name = "MockCtrlInfo02Generator"
    rospy.init_node(node_name)
    rospy.logwarn("Init %s", node_name)
    rate = rospy.Rate(fps)

    msg = Flag_Info()
    msg.Dspace_Flag08 = float(self_driving_mode)

    topic = "/Flag_Info02"
    publisher = rospy.Publisher(topic, Flag_Info, queue_size=1)
    print("publish msg.Dspace_Flag08={} at {} with fps={}".format(msg.Dspace_Flag08, topic, fps))
    while not rospy.is_shutdown():
        publisher.publish(msg)
        rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--self-driving-mode", type=int, default=1,
                        help="1: self-driving, 0: manual-driving")
    args = parser.parse_known_args()[0]
    gen_flag02(args.fps, args.self_driving_mode)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
