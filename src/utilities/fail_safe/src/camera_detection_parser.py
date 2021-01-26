# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import pprint
import datetime
import rospy
from msgs.msg import DetectedObjectArray
from object_ids import OBJECT_ID_TO_NAME


class CameraDetectionParser(object):
    def __init__(self):
        self.msg = None
        topics = ["/cam_obj/back_top_120", "/cam_obj/front_bottom_60",
                  "/cam_obj/front_top_close_120", "/cam_obj/front_top_far_30",
                  "/cam_obj/left_back_60", "/cam_obj/left_front_60",
                  "/cam_obj/right_back_60", "/cam_obj/right_front_60"]
        self.weakest = [(1.0, "") for _ in range(8)]
        for topic in topics:
            rospy.Subscriber(topic, DetectedObjectArray, self._cb)

    def _cb(self, msg):
        for obj in msg.objects:
            assert len(obj.camInfo) == 1
            cam_id = obj.camInfo[0].id
            class_id = obj.classId
            prob = obj.camInfo[0].prob
            if prob < self.weakest[cam_id][0]:
                output = "{}: {} {:.2f}".format(
                    cam_id, OBJECT_ID_TO_NAME[class_id], prob)
                self.weakest[cam_id] = (prob, output)


    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            print("-" * 10 + " " + str(datetime.datetime.now()) + " " + "-" * 10)
            for _, item in self.weakest:
                print(item)
            rate.sleep()

def main():
    rospy.init_node("DetectionParser")
    node = CameraDetectionParser()
    node.run()


if __name__ == "__main__":
    main()
