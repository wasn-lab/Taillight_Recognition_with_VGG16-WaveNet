#!/usr/bin/env python2

import rospy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Point32, Point
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import (
    Header,
    String
)
from rosgraph_msgs.msg import Clock
#


class Node:

    def __init__(self):
        rospy.init_node("text_marker")

        self.inputTopic = rospy.get_param("~topic")
        self.frame_id = rospy.get_param("~frame_id")
        self.c_red = rospy.get_param("~red")
        self.c_green = rospy.get_param("~green")
        self.c_blue = rospy.get_param("~blue")
        self.prefix = rospy.get_param("~prefix", "")
        self.txt_pos_x = rospy.get_param("~txt_pos_x", 3.0)
        self.txt_pos_y = rospy.get_param("~txt_pos_y", 30.0)
        self.t_clock = rospy.Time()
        #
        self.txt_mark_pub = rospy.Publisher(self.inputTopic + "/txt", MarkerArray, queue_size=1)
        #
        # self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.detection_sub = rospy.Subscriber(self.inputTopic, String, self.string_callback)
        #
        self.start_stamp = rospy.get_rostime()
        #
        self.header = Header()
        self.header.frame_id = self.frame_id
        self.header.stamp = self.start_stamp
        #

    def run(self):
        rospy.spin()

    def clock_CB(self, msg):
        self.t_clock = msg.clock

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
        p.x = self.txt_pos_x
        p.y = self.txt_pos_y
        p.z = 2.0
        return p

    def string_callback(self, message):
        current_stamp = rospy.get_rostime()
        elapsed_time = current_stamp - self.start_stamp
        # Update time stamp
        self.header.stamp = current_stamp
        #
        marker_list = MarkerArray()
        marker_list.markers.append( self.create_text_marker( 1, self.header, elapsed_time, message.data, self.text_marker_position_origin() ) )
        self.txt_mark_pub.publish(marker_list)


    def create_text_marker(self, idx, header, elapsed_time, text_in, point):
        """
        Generate a text marker for showing text
        """
        # Generate text
        if len(str(self.prefix)) > 0:
            text = "[%s]" % str(self.prefix)
        else:
            text = ""
        text += "[e=%.1fs] %s" % ( elapsed_time.to_sec(), text_in)
        #
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_T"), scale=2.0 )

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
        marker.lifetime = rospy.Duration(0) # 1.0
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
