#!/usr/bin/env python2

import rospy
from msgs.msg import *


class Node:
    y_thr_min_ = 1.5  # right boundary of left lane (m)
    y_thr_max_ = 5.  # left boundary of left lane (m)

    x_thr1_min_ = -10.
    x_thr1_max_ = 0.

    x_thr2_min_ = 0.
    x_thr2_max_ = 70.

    x_thr3_min_ = -40.
    x_thr3_max_ = -10.

    speed_rel_thr_ = 1.  # km/h

    def __init__(self):
        rospy.init_node("itri_grid_pp")

        self.pub_grid_pp = rospy.Publisher("/lane_event", LaneEvent, queue_size=1)
        self.sub_track3d = rospy.Subscriber("/Tracking3D", DetectedObjectArray, self.callback_track3d)

    def run(self):
        rospy.spin()

    def callback_track3d(self, msg):
        self.pub_grid_pp.publish(self.msg_convert(msg))


    def obj_in_area(self, x_min, x_max, y_min, y_max, x_thr_min, x_thr_max, y_thr_min, y_thr_max):
        is_in = False
        if y_thr_min <= y_min <= y_thr_max or y_thr_min <= y_max <= y_thr_max or (y_min < y_thr_min and y_max > y_thr_max):
            if x_thr_min <= x_min <= x_thr_max or x_thr_min <= x_max <= x_thr_max or (x_min < x_thr_min and x_max > x_thr_max):
                is_in = True
        return is_in

    def msg_convert(self, in_list):
        out = LaneEvent()
        out.header = in_list.header

        out.is_in_n10_0 = False
        out.is_in_0_70_incoming = False
        out.is_in_n40_n10_incoming = False

        out.obj_in_n10_0 = -1
        out.obj_in_0_70_incoming = -1
        out.obj_in_n40_n10_incoming = -1

        for obj in in_list.objects:
            obj_x_min = min(obj.bPoint.p0.x, obj.bPoint.p3.x, obj.bPoint.p4.x, obj.bPoint.p7.x)
            obj_x_max = max(obj.bPoint.p0.x, obj.bPoint.p3.x, obj.bPoint.p4.x, obj.bPoint.p7.x)

            obj_y_min = min(obj.bPoint.p0.y, obj.bPoint.p3.y, obj.bPoint.p4.y, obj.bPoint.p7.y)
            obj_y_max = max(obj.bPoint.p0.y, obj.bPoint.p3.y, obj.bPoint.p4.y, obj.bPoint.p7.y)

            # event 1: is_in_n10_0
            if out.is_in_n10_0 == False:
                if self.obj_in_area(obj_x_min, obj_x_max, obj_y_min, obj_y_max, self.x_thr1_min_, self.x_thr1_max_, self.y_thr_min_, self.y_thr_max_):
                    out.is_in_n10_0 = True
                    out.obj_in_n10_0 = obj.track.id
                    print('Object {0} triggers: is_in_n10_0'.format(obj.track.id % 1000))

            # event 2: is_in_0_70_incoming
            if out.is_in_0_70_incoming == False:
                if obj.relSpeed >= self.speed_rel_thr_ and self.obj_in_area(obj_x_min, obj_x_max, obj_y_min, obj_y_max, self.x_thr2_min_, self.x_thr2_max_, self.y_thr_min_, self.y_thr_max_):
                    out.is_in_0_70_incoming = True
                    out.obj_in_0_70_incoming = obj.track.id
                    print('Object {0} triggers: is_in_0_70_incoming'.format(obj.track.id % 1000))

            # event 3: is_in_n40_n10_incoming
            if out.is_in_n40_n10_incoming == False:
                if obj.relSpeed >= self.speed_rel_thr_ and self.obj_in_area(obj_x_min, obj_x_max, obj_y_min, obj_y_max, self.x_thr3_min_, self.x_thr3_max_, self.y_thr_min_, self.y_thr_max_):
                    out.is_in_n40_n10_incoming = True
                    out.obj_in_n40_n10_incoming = obj.track.id
                    print('Object {0} triggers: is_in_n40_n10_incoming'.format(obj.track.id % 1000))

        return out


if __name__ == "__main__":
    node = Node()
    node.run()
