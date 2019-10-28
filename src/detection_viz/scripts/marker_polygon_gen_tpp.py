#!/usr/bin/env python2

from geometry_msgs.msg import Point
from msgs.msg import DetectedObjectArray
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from rosgraph_msgs.msg import Clock

from msgs.msg import *

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
        self.t_clock = rospy.Time()

        self.box_mark_pub = \
            rospy.Publisher(
                self.inputTopic + "/markers",
                MarkerArray,
                queue_size=1)

        self.delay_txt_mark_pub = \
            rospy.Publisher(
                self.inputTopic + "/delay",
                MarkerArray,
                queue_size=1)

        self.speed_mark_pub = \
            rospy.Publisher(
                self.inputTopic + "/speed",
                MarkerArray,
                queue_size=1)

        self.id_mark_pub = \
            rospy.Publisher(
                self.inputTopic + "/id",
                MarkerArray,
                queue_size=1)

        self.polygon_pub = \
            rospy.Publisher(
                self.inputTopic + "/polygons",
                MarkerArray,
                queue_size=1)

        self.clock_sub = \
            rospy.Subscriber(
                "/clock",
                Clock,
                self.clock_callback)

        self.detection_sub = \
            rospy.Subscriber(
                self.inputTopic,
                DetectedObjectArray,
                self.tpp_callback)


    def clock_callback(self, msg):
        self.t_clock = msg.clock

    def text_marker_position(self, bbox):
        point_1 = PointXYZ()
        point_2 = PointXYZ()
        point_1 = bbox.p1
        point_2 = bbox.p6
        p = Point()
        p.x = (point_1.x + point_2.x) * 0.5
        p.y = (point_1.y + point_2.y) * 0.5
        p.z = (point_1.z + point_2.z) * 0.5 + 2.0
        return p


    def tpp_callback(self, message):
        text_list = MarkerArray()
        text_list2 = MarkerArray()
        text_list3 = MarkerArray()

        box_list = MarkerArray()
        box_list2 = MarkerArray()

        idx = 1
        for i in range(len(message.objects)):
            point = self.text_marker_position(message.objects[i].bPoint)
            text_list3.markers.append( self.create_trackid_marker( idx, message.header, point, message.objects[i].track.id) )
            idx += 1
            text_list.markers.append( self.create_delay_text_marker( idx, message.header, point, message.objects[i].header) )
            idx += 1
            text_list2.markers.append( self.create_speed_marker( idx, message.header, point, message.objects[i].relSpeed) )
            idx += 1
        self.id_mark_pub.publish(text_list3)

        for i in range(len(message.objects)):
            box_list.markers.append( self.create_polygon(message.header, message.objects[i].cPoint, idx) )
            idx += 1
            box_list2.markers.append( self.create_bounding_box_marker( idx, message.header, message.objects[i].bPoint) )
            idx += 1

        self.polygon_pub.publish(box_list)
        self.box_mark_pub.publish(box_list2)

        self.delay_txt_mark_pub.publish(text_list)
        self.speed_mark_pub.publish(text_list2)


    def create_bounding_box_marker(self, idx, header, bbox):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.2
        marker.lifetime = rospy.Duration(0.2)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0

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


    def create_polygon(self, header, cPoint, idx):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.id = idx
        marker.type = Marker.LINE_STRIP
        if len(cPoint.lowerAreaPoints) <= 4:
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.7
        else:
            marker.scale.x = 0.17
            marker.scale.y = 0.17
            marker.scale.z = 0.4
        marker.lifetime = rospy.Duration(0.2)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0

        marker.points = []
        if len(cPoint.lowerAreaPoints) > 0:
            for i in range(len(cPoint.lowerAreaPoints)):
                marker.points.append(cPoint.lowerAreaPoints[i])
            marker.points.append(cPoint.lowerAreaPoints[0])

        return marker


    def create_delay_text_marker(self, idx, header, point, obj_header):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_d"
        marker.action = Marker.ADD
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        # marker.scale.x = 10.0
        # marker.scale.y = 1.0
        marker.scale.z = 1.7
        marker.lifetime = rospy.Duration(0.2)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0
        marker.text = "%.3fms" % ((rospy.get_rostime() - header.stamp).to_sec() * 1000.0)

        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker


    def create_speed_marker(self, idx, header, point, speed):
        # print('relative speed = %.2f' % speed)
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_s"
        marker.action = Marker.ADD
        marker.id = idx
        marker.type=Marker.TEXT_VIEW_FACING
        # marker.scale.x = 10.0
        # marker.scale.y = 1.0
        marker.scale.z = 1.7
        marker.lifetime = rospy.Duration(0.2)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0
        marker.text=str(round(speed, 2))

        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker


    def create_trackid_marker(self, idx, header, point, trackid):
        # print('track id = %d' % trackid)
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_id"
        marker.action = Marker.ADD
        marker.id = idx
        marker.type=Marker.TEXT_VIEW_FACING
        # marker.scale.x = 10.0
        # marker.scale.y = 1.0
        marker.scale.z = 1.7
        marker.lifetime = rospy.Duration(0.2)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0
        marker.text=str(trackid % 1000 + 1)

        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = Node()
    node.run()
