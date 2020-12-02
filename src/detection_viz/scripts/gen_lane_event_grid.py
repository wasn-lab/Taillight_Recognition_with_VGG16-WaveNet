#!/usr/bin/env python2

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from msgs.msg import LaneEvent


class Node:
    y_min_ = 1.5  # right boundary of left lane (m)
    y_max_ = 5.0  # left boundary of left lane (m)

    x1_min_ = 0.0
    x1_max_ = 70.0

    x2_min_ = -10.0
    x2_max_ = 0.0

    x3_min_ = -40.0
    x3_max_ = -10.0

    def __init__(self):
        rospy.init_node("lane_event_grid")

        self.in_topic_ = rospy.get_param("~in_topic")

        self.sub_grid_ = rospy.Subscriber(
            self.in_topic_, LaneEvent, self.callback_lane_event)
        self.pub_grid_ = rospy.Publisher(
            self.in_topic_ + "/grid", Marker, queue_size=1)

    def assign_grid_corners(self, x1, x2, y1, y2):
        p11 = Point()
        p11.x = x1
        p11.y = y1
        p11.z = 0.0

        p12 = Point()
        p12.x = x1
        p12.y = y2
        p12.z = 0.0

        p21 = Point()
        p21.x = x2
        p21.y = y1
        p21.z = 0.0

        p22 = Point()
        p22.x = x2
        p22.y = y2
        p22.z = 0.0

        return p11, p12, p21, p22

    def create_lane_event_grid(self, p11, p12, p21, p22, is_warning):
        points = []
        points.append(p11)
        points.append(p12)
        points.append(p12)
        points.append(p22)
        points.append(p22)
        points.append(p21)
        points.append(p21)
        points.append(p11)

        if is_warning:
            points.append(p11)
            points.append(p22)
            points.append(p12)
            points.append(p21)

        return points

    def create_lane_event_grid_main(self, x1, x2, y1, y2, is_warning):
        p11, p12, p21, p22 = self.assign_grid_corners(x1, x2, y1, y2)
        return self.create_lane_event_grid(p11, p12, p21, p22, is_warning)

    def create_lane_event_grid_list(
            self,
            header,
            is_warning_c1,
            is_warning_c2,
            is_warning_c3):
        marker = Marker()

        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.in_topic_

        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.5
        marker.lifetime = rospy.Duration(1.0)

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = []
        for p in self.create_lane_event_grid_main(
                self.x1_max_,
                self.x1_min_,
                self.y_max_,
                self.y_min_,
                is_warning_c1):
            marker.points.append(p)
        for p in self.create_lane_event_grid_main(
                self.x2_max_,
                self.x2_min_,
                self.y_max_,
                self.y_min_,
                is_warning_c2):
            marker.points.append(p)
        for p in self.create_lane_event_grid_main(
                self.x3_max_,
                self.x3_min_,
                self.y_max_,
                self.y_min_,
                is_warning_c3):
            marker.points.append(p)

        return marker

    def callback_lane_event(self, msg):
        self.pub_grid_.publish(
            self.create_lane_event_grid_list(
                msg.header,
                msg.is_in_0_70_incoming,
                msg.is_in_n10_0,
                msg.is_in_n40_n10_incoming))

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = Node()
    node.run()
