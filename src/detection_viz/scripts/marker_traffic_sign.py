#!/usr/bin/env python

import rospy
from std_msgs.msg import String

from msgs.msg import DetectedSign
from msgs.msg import DetectedSignArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from enum import Enum

from rosgraph_msgs.msg import Clock

SIGN_DICT = {
        0 : 'UNKNOWN',
        1 : 'RAILROAD_CROSSING',
        2 : 'TRAFFIC_SIGNAL',
        3 : 'CLOSED_TO_VEHICLES',
        4 : 'NO_ENTRY',
        5 : 'NO_PARKING',
        6 : 'LEFT_DIRECTION_ONLY',
        7 : 'ONE_WAY', 
        8 : 'SPEED_LIMIT',
        9 : 'SLOW',
        10: 'STOP'
}

class Marker_traffic_sign:
    def __init__(self):
        rospy.init_node('detected_sign_markers')
        self.inputTopic = rospy.get_param("~topic")
        self.light_pub = rospy.Publisher(self.inputTopic + "/markers", MarkerArray, queue_size = 30)
        self.sub_sign = rospy.Subscriber(self.inputTopic, DetectedSignArray, self.convert_sign)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.delay_text_mark_pub = rospy.Publisher(self.inputTopic + "/delay", MarkerArray, queue_size=30)

        self.lifetime = rospy.Duration(1.0)  #for 1 fps update rate

    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def create_traffic_sign_set(self, message, sign_msg):
        markers = []

        # out of range set to unknown
        if sign_msg.classId not in SIGN_DICT:
            marker0.id = 0

        #text
        marker0 = Marker()
        marker0.header.frame_id = message.header.frame_id
        marker0.header.stamp = message.header.stamp
        marker0.ns = self.inputTopic
        marker0.action = Marker.ADD
        marker0.id = sign_msg.classId
        marker0.type = Marker.TEXT_VIEW_FACING
        marker0.scale.z = 1
        marker0.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker0.color.r = 1
        marker0.color.g = 1
        marker0.color.b = 1
        marker0.color.a = 1.0

        marker0.pose.position.x = sign_msg.distance
        marker0.pose.position.y = 3.0
        marker0.pose.position.z = 4.4
        marker0.pose.orientation.w = 1.0

        marker0.text = "dis: %.2fm %s" % ( sign_msg.distance, SIGN_DICT[marker0.id] )
        markers.append(marker0)
        
        #stick
        marker1 = Marker()
        marker1.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker1.header.frame_id = message.header.frame_id
        marker1.header.stamp = message.header.stamp
        marker1.ns = self.inputTopic
        marker1.action = Marker.ADD
        marker1.pose.orientation.w = 1.0
        marker1.id = sign_msg.classId + 20
        marker1.type = Marker.CYLINDER
        marker1.scale.x = 0.2
        marker1.scale.y = 0.2
        marker1.scale.z = 3.6

        marker1.pose.position.x = sign_msg.distance
        marker1.pose.position.y = 3.0
        marker1.pose.position.z = 1.8

        marker1.color.r = 0.8
        marker1.color.g = 0.8
        marker1.color.b = 0.8
        marker1.color.a = 1

        #mark
        marker2 = Marker()
        marker2.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker2.header.frame_id = message.header.frame_id
        marker2.header.stamp = message.header.stamp
        marker2.ns = self.inputTopic
        marker2.action = Marker.ADD
        marker2.id = sign_msg.classId + 40
        marker2.type = Marker.CYLINDER
        marker2.scale.x = 0.8
        marker2.scale.y = 0.8
        marker2.scale.z = 0.1
        marker2.pose.orientation.x = 0.0
        marker2.pose.orientation.y = 1.0
        marker2.pose.orientation.z = 0.0
        marker2.pose.orientation.w = 1.0

        marker2.pose.position.x = sign_msg.distance - 0.15
        marker2.pose.position.y = 3.0
        marker2.pose.position.z = 3.6


        if sign_msg.classId in (1, 2): #these signs are yellow
            marker2.color.r = 1
            marker2.color.g = 1
            marker2.color.b = 0
            marker2.color.a = 1
        elif sign_msg.classId in (3, 4, 5, 8, 9, 10): #these signs are red
            marker2.color.r = 1
            marker2.color.g = 0
            marker2.color.b = 0
            marker2.color.a = 1
        elif sign_msg.classId in (6, 7): #these signs are blue
            marker2.color.r = 0
            marker2.color.g = 0
            marker2.color.b = 1
            marker2.color.a = 1
        else:
            marker2.color.r = 0.8
            marker2.color.g = 0.8
            marker2.color.b = 0.8
            marker2.color.a = 1

        markers.append(marker1)
        markers.append(marker2)
    
        return markers

    def create_delay_text_marker(self, idx, header, dis):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_d"
        marker.action = Marker.ADD
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.scale.z = 1
        marker.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1.0
        marker.text = "%.3fms" % ((rospy.get_rostime() - header.stamp).to_sec() * 1000.0)

        marker.pose.position.x = dis
        marker.pose.position.y = 3.0
        marker.pose.position.z = 5.2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker

    def convert_sign(self, message):
        #print("CALL ONE!")
        marker_list = MarkerArray()
        text_list = MarkerArray()
        idx = 0

        for sign_msg in message.signs:
            marker_list.markers.extend(self.create_traffic_sign_set(message, sign_msg))
            text_list.markers.append( self.create_delay_text_marker( idx, message.header, sign_msg.distance))
            idx += 1
        self.light_pub.publish(marker_list)
        self.delay_text_mark_pub.publish(text_list)


    def run(self):
        print("RUN!")
        rospy.spin()


if __name__ == '__main__':
    node = Marker_traffic_sign()
    node.run()

