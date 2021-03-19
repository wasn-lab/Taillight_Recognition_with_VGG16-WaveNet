#!/usr/bin/env python2

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from msgs.msg import LaneEvent


class Node:
    ground_z = -3.1

    x1_min_ = 1.0
    x1_max_ = 71.0
    x1_center_ = (x1_min_ + x1_max_) / 2
    x1_scale_ = x1_max_ - x1_min_

    y1_min_ = 1.5  # right boundary of left lane (m)
    y1_max_ = 5.0  # left boundary of left lane (m)
    y1_center_ = (y1_min_ + y1_max_) / 2
    y1_scale_ = y1_max_ - y1_min_

    x2_min_ = -9.0
    x2_max_ = 1.0
    x2_center_ = (x2_min_ + x2_max_) / 2
    x2_scale_ = x2_max_ - x2_min_

    y2_min_ = -1.5  # right boundary of left lane (m)
    y2_max_ = 5.0  # left boundary of left lane (m)
    y2_center_ = (y2_min_ + y2_max_) / 2
    y2_scale_ = y2_max_ - y2_min_

    x3_min_ = -39.0
    x3_max_ = -9.0
    x3_center_ = (x3_min_ + x3_max_) / 2
    x3_scale_ = x3_max_ - x3_min_

    y3_min_ = -1.5  # right boundary of left lane (m)
    y3_max_ = 5.0  # left boundary of left lane (m)
    y3_center_ = (y3_min_ + y3_max_) / 2
    y3_scale_ = y3_max_ - y3_min_

    def __init__(self):
        rospy.init_node("lane_event_grid")

        self.in_topic_ = rospy.get_param("~in_topic")

        self.sub_grid_ = rospy.Subscriber(
            self.in_topic_, LaneEvent, self.callback_lane_event)
        self.pub_grid_ = rospy.Publisher(
            self.in_topic_ + "/grid", Marker, queue_size=1)
        self.pub_c1_ = rospy.Publisher(
            self.in_topic_ + "/c1", Marker, queue_size=1)
        self.pub_c2_ = rospy.Publisher(
            self.in_topic_ + "/c2", Marker, queue_size=1)
        self.pub_c3_ = rospy.Publisher(
            self.in_topic_ + "/c3", Marker, queue_size=1)

    def assign_grid_corners(self, x1, x2, y1, y2):
        p11 = Point()
        p11.x = x1
        p11.y = y1
        p11.z = self.ground_z

        p12 = Point()
        p12.x = x1
        p12.y = y2
        p12.z = self.ground_z

        p21 = Point()
        p21.x = x2
        p21.y = y1
        p21.z = self.ground_z

        p22 = Point()
        p22.x = x2
        p22.y = y2
        p22.z = self.ground_z

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

        return points

    def create_lane_event_grid_main(self, x1, x2, y1, y2, is_warning):
        p11, p12, p21, p22 = self.assign_grid_corners(x1, x2, y1, y2)
        return self.create_lane_event_grid(p11, p12, p21, p22, is_warning)

    def create_lane_event_grid_list(
            self,
            header,
            idx,
            is_warning_c1,
            is_warning_c2,
            is_warning_c3):
        marker = Marker()

        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.in_topic_

        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.4
        marker.lifetime = rospy.Duration(1.0)

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.points = []
        for p in self.create_lane_event_grid_main(
                self.x1_max_,
                self.x1_min_,
                self.y1_max_,
                self.y1_min_,
                is_warning_c1):
            pass
            # marker.points.append(p)
        for p in self.create_lane_event_grid_main(
                self.x2_max_,
                self.x2_min_,
                self.y2_max_,
                self.y2_min_,
                is_warning_c2):
            marker.points.append(p)
        for p in self.create_lane_event_grid_main(
                self.x3_max_,
                self.x3_min_,
                self.y3_max_,
                self.y3_min_,
                is_warning_c3):
            marker.points.append(p)

        return marker

    def create_lane_event_grid_warning_list(
            self,
            header,
            idx,
            x,
            y,
            z,
            scale_x,
            scale_y):
        marker = Marker()

        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.in_topic_

        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = 0.1
        marker.lifetime = rospy.Duration(1.0)

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.3

        return marker

    def callback_lane_event(self, msg):
        idx = 0
        self.pub_grid_.publish(
            self.create_lane_event_grid_list(
                msg.header,
                idx,
                msg.is_in_0_70_incoming,
                msg.is_in_n10_0,
                msg.is_in_n40_n10_incoming))

        idx += 1
        if msg.is_in_0_70_incoming:
            self.pub_c1_.publish(
                self.create_lane_event_grid_warning_list(
                    msg.header,
                    idx,
                    self.x1_center_,
                    self.y1_center_,
                    self.ground_z,
                    self.x1_scale_,
                    self.y1_scale_,
                ))

        idx += 1
        if msg.is_in_n10_0:
            self.pub_c2_.publish(
                self.create_lane_event_grid_warning_list(
                    msg.header,
                    idx,
                    self.x2_center_,
                    self.y2_center_,
                    self.ground_z,
                    self.x2_scale_,
                    self.y2_scale_,
                ))

        idx += 1
        if msg.is_in_n40_n10_incoming:
            self.pub_c3_.publish(
                self.create_lane_event_grid_warning_list(
                    msg.header,
                    idx,
                    self.x3_center_,
                    self.y3_center_,
                    self.ground_z,
                    self.x3_scale_,
                    self.y3_scale_,
                ))

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = Node()
    node.run()
