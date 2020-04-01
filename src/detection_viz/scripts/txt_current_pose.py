#!/usr/bin/env python2

import copy
import rospy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Point32, Point
# from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
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
        self.delay_pos_y = rospy.get_param("~delay_pos_y", 30.0)
        self.txt_frame_id = rospy.get_param("~txt_frame_id", "txt_frame")
        self.t_clock = rospy.Time()
        #
        self.delay_txt_mark_pub = rospy.Publisher(self.inputTopic + "/delayTxt", MarkerArray, queue_size=1)
        #
        # self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.detection_sub = rospy.Subscriber(self.inputTopic, PoseStamped, self.detection_callback)
        # FPS
        self.fps_cal = FPS.FPS()
        # Flags
        self.is_overwrite_txt_frame_id = (len(self.txt_frame_id) != 0)

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
        p.y = self.delay_pos_y
        p.z = 2.0
        return p

    def detection_callback(self, message):
        current_stamp = rospy.get_rostime()
        self.fps_cal.step()
        # print("fps = %f" % self.fps_cal.fps)
        delay_list = MarkerArray()
        delay_list.markers.append( self.create_delay_text_marker( 1, message.header, current_stamp, self.text_marker_position_origin(), self.fps_cal.fps ) )
        #
        self.delay_txt_mark_pub.publish(delay_list)


    def create_delay_text_marker(self, idx, header, current_stamp, point, fps=None):
        # Overwrite the frame_id of the text
        header_txt = copy.deepcopy(header)
        if self.is_overwrite_txt_frame_id:
            header_txt.frame_id = self.txt_frame_id
        else:
            header_txt.frame_id = "lidar"
        #
        marker = Marker()
        marker.header.frame_id = header_txt.frame_id
        marker.header.stamp = header_txt.stamp
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
        # marker.text = "%.3fms" % ((rospy.get_rostime() - header_txt.stamp).to_sec() * 1000.0)
        if len(str(self.delay_prefix)) > 0:
            marker.text = "[%s] " % str(self.delay_prefix)
        else:
            marker.text = ""
        marker.text += "%.3fms" % ((current_stamp - header_txt.stamp).to_sec() * 1000.0)
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
