#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import sys
import rospy
from msgs.msg import Flag_Info


def main():
    node_name = "FakeFlagInfo02"
    rospy.init_node(node_name)
    rospy.logwarn("Init %s", node_name)
    rate = rospy.Rate(10)  # FPS: 10

    msg = Flag_Info()
    msg.Dspace_Flag08 = 1.0

    publisher = rospy.Publisher("/Flag_Info02", Flag_Info, queue_size=1)
    print("Keep publishing msg.Dspace_Flag08 = {}".format(msg.Dspace_Flag08))
    while not rospy.is_shutdown():
        publisher.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
