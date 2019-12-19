#!/usr/bin/env python2


import rospy
from std_msgs.msg import (
    Bool,
)
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Point
from msgs.msg import DetectedObjectArray
from rosgraph_msgs.msg import Clock
#
import numpy as np
import fps_calculator as FPS

BOX_ORDER = [
    0, 1,
    1, 2,
    2, 3,
    3, 0,

    4, 5,
    5, 6,
    6, 7,
    7, 4,

    0, 4,
    1, 5,
    2, 6,
    3, 7
]



class Node:

    def __init__(self):
        rospy.init_node("detected_object_markers")
        self.inputTopic = rospy.get_param("~topic")
        self.c_red = rospy.get_param("~red")
        self.c_green = rospy.get_param("~green")
        self.c_blue = rospy.get_param("~blue")
        self.delay_prefix = rospy.get_param("~delay_prefix", "")
        self.delay_pos_x = rospy.get_param("~delay_pos_x", 3.0)
        self.delay_pos_y = rospy.get_param("~delay_pos_y", 30.0)
        self.t_clock = rospy.Time()

        self.box_mark_pub = rospy.Publisher(self.inputTopic + "/bbox", MarkerArray, queue_size=1)
        self.delay_txt_mark_pub = rospy.Publisher(self.inputTopic + "/delayTxt", MarkerArray, queue_size=1)

        # self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.detection_sub = rospy.Subscriber(self.inputTopic, DetectedObjectArray, self.detection_callback)
        self.is_showing_depth_sub = rospy.Subscriber("/d_viz/req_show_depth", Bool, self.req_show_depth_CB)
        # FPS
        self.fps_cal = FPS.FPS()
        # Depth
        self.is_showing_depth = True

    def run(self):
        rospy.spin()

    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def req_show_depth_CB(self, msg):
        self.is_showing_depth = msg.data

    def text_marker_position(self, bbox):
        point_1 = bbox.p1
        point_2 = bbox.p6
        p = Point()
        p.x = (point_1.x + point_2.x) * 0.5 + 2.0
        p.y = (point_1.y + point_2.y) * 0.5
        p.z = (point_1.z + point_2.z) * 0.5
        return p

    def text_marker_position_origin(self):
        p = Point()
        p.x = self.delay_pos_x
        p.y = self.delay_pos_y
        p.z = 2.0
        return p

    def _calculate_depth_bbox(self, bbox):
        """
        The depth of a bbox is simply the x value of p0.
        """
        return abs(bbox.p0.x)

    def _calculate_distance_bbox(self, bbox):
        """
        The distance of a bbox is the Euclidean distance between origin and (p0+p4)/2.
        """
        point_1 = np.array( (bbox.p0.x, bbox.p0.y) )
        point_2 = np.array( (bbox.p4.x, bbox.p4.y) )
        return (0.5 * np.linalg.norm( (point_1 + point_2) ) )

    def detection_callback(self, message):
        current_stamp = rospy.get_rostime()
        self.fps_cal.step()
        # print("fps = %f" % self.fps_cal.fps)
        box_list = MarkerArray()
        delay_list = MarkerArray()
        box_list.markers.append(self.create_bounding_box_list_marker(1, message.header, message.objects ) )
        delay_list.markers.append( self.create_delay_text_marker( 1, message.header, current_stamp, self.text_marker_position_origin(), self.fps_cal.fps ) )
        # idx = 1
        # for i in range(len(message.objects)):
        #     # point = self.text_marker_position(message.objects[i].bPoint)
        #     box_list.markers.append( self.create_bounding_box_marker( idx, message.header, message.objects[i].bPoint) )
        #     # delay_list.markers.append( self.create_delay_text_marker( idx, message.header, point) )
        #     idx += 1
        if self.is_showing_depth:
            idx = 2
            for i in range(len(message.objects)):
                box_list.markers.append( self.create_depth_text_marker( idx, message.header, message.objects[i].bPoint, i) )
                idx += 1
        #
        self.box_mark_pub.publish(box_list)
        self.delay_txt_mark_pub.publish(delay_list)


    # def create_bounding_box_marker(self, idx, header, bbox):
    #     marker = Marker()
    #     marker.header.frame_id = header.frame_id
    #     marker.header.stamp = header.stamp
    #     marker.ns = self.inputTopic
    #     marker.action = Marker.ADD
    #     marker.pose.orientation.w = 1.0
    #     marker.id = idx
    #     marker.type = Marker.LINE_LIST
    #     marker.scale.x = 0.2
    #     marker.lifetime = rospy.Duration(1.0)
    #     marker.color.r = self.c_red
    #     marker.color.g = self.c_green
    #     marker.color.b = self.c_blue
    #     marker.color.a = 1.0
    #
    #     point_list = [
    #         bbox.p0,
    #         bbox.p1,
    #         bbox.p2,
    #         bbox.p3,
    #         bbox.p4,
    #         bbox.p5,
    #         bbox.p6,
    #         bbox.p7
    #     ]
    #
    #     for index in BOX_ORDER:
    #         point = point_list[index]
    #         point_msg = Point()
    #         point_msg.x = point.x
    #         point_msg.y = point.y
    #         point_msg.z = point.z
    #         marker.points.append(point_msg)
    #
    #     return marker

    def create_bounding_box_list_marker(self, idx, header, objects):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.2
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0


        for _i in range(len(objects)):
            bbox = objects[_i].bPoint
            point_list = [
                bbox.p0,
                bbox.p1,
                bbox.p2,
                bbox.p3,
                bbox.p4,
                bbox.p5,
                bbox.p6,
                bbox.p7
            ]

            for index in BOX_ORDER:
                point = point_list[index]
                point_msg = Point()
                point_msg.x = point.x
                point_msg.y = point.y
                point_msg.z = point.z
                marker.points.append(point_msg)

        return marker


    def create_delay_text_marker(self, idx, header, current_stamp, point, fps=None):
        """
        Generate a text marker for showing latency and FPS.
        """
        # Generate text
        if len(str(self.delay_prefix)) > 0:
            text = "[%s] " % str(self.delay_prefix)
        else:
            text = ""
        text += "%.3fms" % ((current_stamp - header.stamp).to_sec() * 1000.0)
        if not fps is None:
            text += " fps = %.1f" % fps
        #
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_delay"), scale=2.0 )

    def create_depth_text_marker(self, idx, header, bbox, bbox_id=None):
        """
        Generate a text marker for showing latency and FPS.
        """
        point = self.text_marker_position( bbox )
        # depth = self._calculate_depth_bbox( bbox )
        depth = self._calculate_distance_bbox( bbox )
        # Generate text
        if bbox_id is None:
            text = "D=%.2fm" % ( depth )
        else:
            text = "[%d]D=%.2fm" % (bbox_id, depth )
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_depth"), scale=2.0 )


    def text_marker_prototype(self, idx, header, text, point=Point(), ns="T", scale=2.0):
        """
        Generate the prototype of text
        """
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = ns
        marker.action = Marker.ADD
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        # marker.scale.x = 10.0
        # marker.scale.y = 1.0
        marker.scale.z = scale
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
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


if __name__ == "__main__":
    node = Node()
    node.run()
