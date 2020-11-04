#!/usr/bin/env python
import argparse
import rospy
from std_msgs.msg import Bool

class MockBackendConnectionGenerator():
    """Use to generate fake backend connection message"""
    def __init__(self, connected, fps):
        rospy.init_node("MockBackendConnectionGenerator")
        self.connected = connected
        self.fps = fps;
        self.publisher = rospy.Publisher("/backend/connected", Bool, queue_size=1)

    def run(self):
        rate = rospy.Rate(self.fps)
        cnt = 0
        while not rospy.is_shutdown():
            if cnt == 0:
                rospy.logwarn("publish mock /backend/connected")
            cnt += 1
            if cnt == 100:
                cnt = 0
            msg = Bool()
            msg.data = self.connected
            self.publisher.publish(msg)
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connected", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_known_args()[0]
    gnr = MockBackendConnectionGenerator(args.connected, args.fps)
    gnr.run()


if __name__ == "__main__":
    main()
