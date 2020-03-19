#!/usr/bin/env python
"""
Calculate the publishing hz of given topics.
"""
from __future__ import print_function
import argparse
import time
import logging
import rospy
# other msg types: CompressedImage, PointCloud2
from sensor_msgs.msg import Image, PointCloud2

class HZCalculatorNode(object):
    def __init__(self, duration):
        self.duration = duration
        logging.warning("Listen for %d seconds to calculate topic frequency.", duration)

        rospy.init_node("HZCalculatorNode")
        self.subscriptions = [
            # for car model B1
            {"topic": "/cam/F_center",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/F_left/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/F_center/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/F_right/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/R_front/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/R_rear/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/L_front/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/L_rear/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/F_top/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/B_top/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            # for car model B1_V2
            {"topic": "/cam/front_bottom_60",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/front_bottom_60/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/front_top_far_30/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/front_top_close_120/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/right_front_60/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/right_back_60/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/left_front_60/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/left_back_60/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/cam/back_top_120/detect_image",
             "type": Image,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/LidarAll",
             "type": PointCloud2,
             "callback": self._callback,
             "counts": 0}]

        for sub in self.subscriptions:
            rospy.Subscriber(sub["topic"], sub["type"], sub["callback"], sub)
            print("Subscribe {}".format(sub["topic"]))

    def _callback(self, __msg, sub):
        if sub["counts"] == 0:
            sub["first_receive_time"] = time.time()
        sub["counts"] += 1

    def run(self):
        rate = rospy.Rate(30)
        done = False
        start_time = time.time()
        while not rospy.is_shutdown() and (not done):
            time_span = time.time() - start_time
            if time_span > self.duration:
                done = True
            else:
                rate.sleep()

        for sub in self.subscriptions:
            now = time.time()
            span = now - sub.get("first_receive_time", now)
            if span > 0:
                _hz = sub["counts"] / span
                print("{} Hz: {:.2f}".format(sub["topic"], _hz))
            else:
                print("{} Hz: No data".format(sub["topic"]))

def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", "-u", type=int, default=60,
                        help="(in seconds) the sampling duration")
    args, __unknown = parser.parse_known_args()

    node = HZCalculatorNode(args.duration)
    node.run()

if __name__ == '__main__':
    main()
