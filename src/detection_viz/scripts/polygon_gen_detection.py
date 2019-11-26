#!/usr/bin/env python2

import rospy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Point
from std_msgs.msg import Header
from msgs.msg import *
from rosgraph_msgs.msg import Clock
#
import fps_calculator as FPS

class Node:

    def __init__(self):
        rospy.init_node("detected_object_polygons")

        self.inputTopic = rospy.get_param("~topic")
        self.c_red = rospy.get_param("~red")
        self.c_green = rospy.get_param("~green")
        self.c_blue = rospy.get_param("~blue")
        self.delay_prefix = rospy.get_param("~delay_prefix", "")
        self.delay_pos_x = rospy.get_param("~delay_pos_x", 3.0)
        self.t_clock = rospy.Time()

        self.polygon_pub = rospy.Publisher(self.inputTopic + "/poly", MarkerArray, queue_size=1)
        self.delay_txt_mark_pub = rospy.Publisher(self.inputTopic + "/delayTxt", MarkerArray, queue_size=1)

        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.detection_sub = rospy.Subscriber(self.inputTopic, DetectedObjectArray, self.detection_callback)
        # FPS
        self.fps_cal = FPS.FPS()

    def run(self):
        rospy.spin()

    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def text_marker_position(self, cPoint):
        point_1 = cPoint.lowerAreaPoints[0]
        p = Point()
        p.x = point_1.x
        p.y = point_1.y
        p.z = point_1.z + 2.0
        return p

    def text_marker_position_origin(self):
        p = Point()
        p.x = self.delay_pos_x
        p.y = 0.0
        p.z = 2.0
        return p

    def detection_callback(self, message):
        current_stamp = rospy.get_rostime()
        self.fps_cal.step()
        # print("fps = %f" % self.fps_cal.fps)
        box_list = MarkerArray()
        delay_list = MarkerArray()
        idx = 1
        for i in range(len(message.objects)):
            # point = self.text_marker_position(message.objects[i].cPoint)
            box_list.markers.append(self.create_polygon(message.header, message.objects[i].cPoint, idx))
            idx += 1
        #
        delay_list.markers.append( self.create_delay_text_marker( 1, message.header, current_stamp, self.text_marker_position_origin(), self.fps_cal.fps ) )
        #
        self.polygon_pub.publish(box_list)
        self.delay_txt_mark_pub.publish(delay_list)


    def create_polygon(self, header, cPoint, idx):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.id = idx
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.1
        marker.lifetime = rospy.Duration(1.0)
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

    def create_delay_text_marker(self, idx, header, current_stamp, point, fps=None):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_d"
        marker.action = Marker.ADD
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        # marker.scale.x = 10.0
        # marker.scale.y = 1.0
        marker.scale.z = 2.0
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0
        # marker.text = "%.3fms" % ((rospy.get_rostime() - header.stamp).to_sec() * 1000.0)
        if len(str(self.delay_prefix)) > 0:
            marker.text = "[%s] " % str(self.delay_prefix)
        else:
            marker.text = ""
        marker.text += "%.3fms" % ((current_stamp - header.stamp).to_sec() * 1000.0)
        if not fps is None:
            marker.text += " fps = %.1f" % fps

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
