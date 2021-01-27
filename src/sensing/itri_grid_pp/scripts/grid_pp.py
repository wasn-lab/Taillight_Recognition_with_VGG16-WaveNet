#!/usr/bin/env python2

import rospy
from msgs.msg import *
from std_msgs.msg import Header
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point


class Node:
    x_thr1_min_ = 1.0
    x_thr1_max_ = 71.0

    y_thr1_min_ = 1.5  # right boundary of left lane (m)
    y_thr1_max_ = 5.0  # left boundary of left lane (m)

    x_thr2_min_ = -9.0
    x_thr2_max_ = 1.0

    y_thr2_min_ = -1.5  # right boundary of host lane (m)
    y_thr2_max_ = 5.0  # left boundary of left lane (m)

    x_thr3_min_ = -39.0
    x_thr3_max_ = -9.0

    y_thr3_min_ = -1.5  # right boundary of host lane (m)
    y_thr3_max_ = 5.0  # left boundary of left lane (m)

    speed_rel_thr_ = -1.0  # km/h; negative value represents 'incoming'

    frame_id_target = "base_link"

    def __init__(self):
        rospy.init_node("itri_grid_pp")

        self.tracking_topic = rospy.get_param("~tracking_topic")
        self.radar_topic = rospy.get_param("~radar_topic")
        self.out_topic = rospy.get_param("~out_topic")

        self.pub_grid_pp = rospy.Publisher(
            self.out_topic, LaneEvent, queue_size=1)
        self.pub_grid_pp_signal = rospy.Publisher(
            self.out_topic + "/signal", MarkerArray, queue_size=1)
        self.sub_track3d = rospy.Subscriber(
            self.tracking_topic,
            DetectedObjectArray,
            self.callback_track3d)
        self.sub_radar = rospy.Subscriber(
            self.radar_topic,
            DetectedObjectArray,
            self.callback_track3d)

    def run(self):
        rospy.spin()

    def callback_track3d(self, msg):
        lane_event = self.msg_convert(msg)
        self.pub_grid_pp.publish(lane_event)
        self.pub_grid_pp_signal.publish(
            self.gen_signal_txt(
                msg.header, lane_event))

    def text_marker_prototype(
            self,
            idx,
            header,
            text,
            point=Point(),
            scale=4.0,
            ns="T"):
        """
        Generate the prototype of text
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id_target
        marker.header.stamp = header.stamp
        marker.ns = ns
        marker.action = Marker.ADD
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.scale.z = scale
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = 0.87451
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.text = text

        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        return marker

    def gen_signal_txt(self, header, lane_event):
        out = MarkerArray()

        txt = ""

        if lane_event.is_in_0_70_incoming:
            txt = txt + "C1:T\n"
            # txt = txt + "C1:T(" + str(lane_event.obj_in_0_70_incoming % 1000) + ")\n"
        else:
            txt = txt + "C1:F\n"

        if lane_event.is_in_n10_0:
            txt = txt + "C2:T\n"
            # txt = "C2:T(" + str(lane_event.obj_in_n10_0 % 1000) + ")\n"
        else:
            txt = txt + "C2:F\n"

        if lane_event.is_in_n40_n10_incoming:
            txt = txt + "C3:T\n"
            # txt = txt + "C3:T(" + str(lane_event.obj_in_n40_n10_incoming % 1000) + ")\n"
        else:
            txt = txt + "C3:F"

        out.markers.append(self.text_marker_prototype(
            0, header, txt, Point(-3, -7, 0)))

        return out

    def obj_in_area(
            self,
            x_min,
            x_max,
            y_min,
            y_max,
            x_thr_min,
            x_thr_max,
            y_thr_min,
            y_thr_max):
        is_in = False
        if y_thr_min <= y_min <= y_thr_max or y_thr_min <= y_max <= y_thr_max or (
                y_min < y_thr_min and y_max > y_thr_max):
            if x_thr_min <= x_min <= x_thr_max or x_thr_min <= x_max <= x_thr_max or (
                    x_min < x_thr_min and x_max > x_thr_max):
                is_in = True
        return is_in

    def msg_convert(self, in_list):
        out = LaneEvent()
        out.header = in_list.header
        out.header.frame_id = self.frame_id_target

        out.is_in_n10_0 = False
        out.is_in_0_70_incoming = False
        out.is_in_n40_n10_incoming = False

        out.obj_in_n10_0 = -1
        out.obj_in_0_70_incoming = -1
        out.obj_in_n40_n10_incoming = -1

        for obj in in_list.objects:
            obj_x_min = min(
                obj.bPoint.p0.x,
                obj.bPoint.p3.x,
                obj.bPoint.p4.x,
                obj.bPoint.p7.x)
            obj_x_max = max(
                obj.bPoint.p0.x,
                obj.bPoint.p3.x,
                obj.bPoint.p4.x,
                obj.bPoint.p7.x)

            obj_y_min = min(
                obj.bPoint.p0.y,
                obj.bPoint.p3.y,
                obj.bPoint.p4.y,
                obj.bPoint.p7.y)
            obj_y_max = max(
                obj.bPoint.p0.y,
                obj.bPoint.p3.y,
                obj.bPoint.p4.y,
                obj.bPoint.p7.y)

            # event C1: is_in_0_70_incoming
            if not out.is_in_0_70_incoming:
                if obj.speed_rel <= -self.speed_rel_thr_ and self.obj_in_area(
                        obj_x_min,
                        obj_x_max,
                        obj_y_min,
                        obj_y_max,
                        self.x_thr1_min_,
                        self.x_thr1_max_,
                        self.y_thr1_min_,
                        self.y_thr1_max_):
                    out.is_in_0_70_incoming = True
                    out.obj_in_0_70_incoming = obj.track.id
                    print(
                        'Object {0} triggers: is_in_0_70_incoming'.format(
                            obj.track.id %
                            1000))

            # event C2: is_in_n10_0
            if not out.is_in_n10_0:
                if self.obj_in_area(
                        obj_x_min,
                        obj_x_max,
                        obj_y_min,
                        obj_y_max,
                        self.x_thr2_min_,
                        self.x_thr2_max_,
                        self.y_thr2_min_,
                        self.y_thr2_max_):
                    out.is_in_n10_0 = True
                    out.obj_in_n10_0 = obj.track.id
                    print(
                        'Object {0} triggers: is_in_n10_0'.format(
                            obj.track.id %
                            1000))

            # event C3: is_in_n40_n10_incoming
            if not out.is_in_n40_n10_incoming:
                if obj.speed_rel <= self.speed_rel_thr_ and self.obj_in_area(
                        obj_x_min,
                        obj_x_max,
                        obj_y_min,
                        obj_y_max,
                        self.x_thr3_min_,
                        self.x_thr3_max_,
                        self.y_thr3_min_,
                        self.y_thr3_max_):
                    out.is_in_n40_n10_incoming = True
                    out.obj_in_n40_n10_incoming = obj.track.id
                    print(
                        'Object {0} triggers: is_in_n40_n10_incoming'.format(
                            obj.track.id %
                            1000))

        return out


if __name__ == "__main__":
    node = Node()
    node.run()
