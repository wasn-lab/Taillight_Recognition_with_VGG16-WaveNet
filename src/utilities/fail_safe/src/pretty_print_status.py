# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import json
import pprint
import rospy
from std_msgs.msg import String

_TOPIC = "/vehicle/report/itri/fail_safe_status"


class PrettyPrintStatus(object):
    def __init__(self):
        rospy.init_node("PrettyPrintStatus", anonymous=True)
        rospy.logwarn("Init PrettyPrintStatus")
        rospy.wait_for_message(_TOPIC, String)
        rospy.Subscriber(_TOPIC, String, self._cb)

    def _cb(self, msg):
        jdata = json.loads(msg.data)
        pprint.pprint(jdata)

    def run(self):
        """Send out aggregated info to backend server every second."""
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    printer = PrettyPrintStatus()
    printer.run()
