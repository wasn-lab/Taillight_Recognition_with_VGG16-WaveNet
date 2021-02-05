#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import argparse
import rospy
from std_msgs.msg import Bool

class MockRecorderGenerator():
    """Use to generate fake recorder message"""
    def __init__(self, fps):
        rospy.init_node("MockRecorderGenerator")
        self.fps = fps;
        self.publisher = rospy.Publisher("/REC/is_recording", Bool, queue_size=1, latch=True)

    def run(self):
        rate = rospy.Rate(self.fps)
        rospy.logwarn("publish mock /REC/is_recording")
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)
        while not rospy.is_shutdown():
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_known_args()[0]
    gnr = MockRecorderGenerator(args.fps)
    gnr.run()


if __name__ == "__main__":
    main()
