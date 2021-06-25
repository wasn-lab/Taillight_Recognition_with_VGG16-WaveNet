# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import argparse
import json
import rospy
from std_msgs.msg import String


class PrettyPrintStatus(object):
    def __init__(self, topic):
        rospy.init_node("PrettyPrintStatus", anonymous=True)
        rospy.logwarn("Init PrettyPrintStatus")
        rospy.wait_for_message(topic, String)
        rospy.Subscriber(topic, String, self._cb)

    def _cb(self, msg):
        jdata = json.loads(msg.data)
        print(json.dumps(jdata, indent=2))

    def run(self):
        """Send out aggregated info to backend server every second."""
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", "-t", default="/vehicle/report/itri/fail_safe_status")
    args = parser.parse_args()
    printer = PrettyPrintStatus(args.topic)
    printer.run()

if __name__ == "__main__":
    main()
