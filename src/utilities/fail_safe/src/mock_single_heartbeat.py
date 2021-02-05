#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import argparse
import rospy
from std_msgs.msg import Empty

class MockSingleHeartbeatGenerator():
    def __init__(self, topic_name, fps):
        rospy.init_node("MockSingleHeartbeatGenerator")
        self.fps = fps
        self.topic_name = topic_name
        self.publisher = rospy.Publisher(topic_name, Empty, queue_size=1)

    def run(self):
        rate = rospy.Rate(self.fps)
        rospy.logwarn("publish heartbeat at %s @%d fps", self.topic_name, self.fps)
        while not rospy.is_shutdown():
            self.publisher.publish()
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-name", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_known_args()[0]
    gnr = MockSingleHeartbeatGenerator(args.topic_name, args.fps)

    gnr.run()


if __name__ == "__main__":
    main()
