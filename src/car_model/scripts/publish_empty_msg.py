#!/usr/bin/env python
"""
Publish an empty message after the specified dealy.
This script is used with address sanitizer to run a program for a period.
"""
from __future__ import print_function
import argparse
import time
import rospy
from std_msgs.msg import Empty


def publish_after_delay(topic, delay):
    rospy.init_node("EmptyMessagePublisher")
    rospy.logwarn("Publish %s after %d seconds", topic, delay)
    while delay > 0:
        time.sleep(1)
        # keep printing out message to prevent misjudged as timeout.
        rospy.logwarn("Count down value: %d", delay)
        delay -= 1
    publisher = rospy.Publisher(topic, Empty, queue_size=1)
    publisher.publish(Empty())
    rospy.logwarn("Done publishing %s", topic)


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", "-t", required=True, help="Topic name")
    parser.add_argument("--delay", "-d", required=True, type=int, default=300)
    args, __unknown = parser.parse_known_args()
    publish_after_delay(args.topic, args.delay)


if __name__ == '__main__':
    main()
