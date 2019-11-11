"""
Calculate the publishing hz of given topics.
"""
from __future__ import print_function
import pprint
import rospy
import time
from sensor_msgs.msg import CompressedImage, PointCloud2

class HZCalculatorNode(object):
    def __init__(self):
        print("Listen for 1 minute and calculate topic frequency.")
        rospy.init_node("HZCalculatorNode")
        self.subscriptions = [
            {"topic": "/gmsl_camera/port_d/cam_0/image_raw/compressed",
             "type": CompressedImage,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/gmsl_camera/port_a/cam_0/image_raw/compressed",
             "type": CompressedImage,
             "callback": self._callback,
             "counts": 0},
            {"topic": "/LidarAll",
             "type": PointCloud2,
             "callback": self._callback,
             "counts": 0}]

        for sub in self.subscriptions:
            rospy.Subscriber(sub["topic"], sub["type"], sub["callback"], sub)
            print("Subscribe {}".format(sub["topic"]))

    def _callback(self, msg, sub):
        if sub["counts"] == 0:
            sub["first_receive_time"] = time.time()
        sub["counts"] += 1

    def run(self):
        rate = rospy.Rate(60)
        done = False
        start_time = time.time()
        time_span = 60
        while not rospy.is_shutdown() and (not done):
            now = time.time()
            time_span = now - start_time
            if time_span > 60:
                done = True
            else:
                rate.sleep()
        for sub in self.subscriptions:
            span = now - sub.get("first_receive_time", now)
            if span == 0:
                hz = 0
            else:
                hz = sub["counts"]/span
            print("{} Hz: {:.2f}".format(sub["topic"], hz))

def main():
    """Prog entry"""
    node = HZCalculatorNode()
    node.run()

if __name__ == '__main__':
    main()
