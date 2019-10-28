#!/usr/bin/env python

import rospy
from std_msgs.msg import String

from msgs.msg import DetectedLight
from msgs.msg import DetectedLightArray
from msgs.msg import DetectedSign
from msgs.msg import DetectedSignArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from enum import Enum

from rosgraph_msgs.msg import Clock

'''
SIGNAL_DICT = { SIGNAL_UNKNOWN      : 0,
                SIGNAL_RED          : 1, 
                SIGNAL_YELLOW       : 2,
                SIGNAL_GREEN        : 3,
                SIGNAL_RED_RIGHT    : 4,
                SIGNAL_PED_GO       : 5,
                SIGNAL_PED_STOP     : 6,
                SIGNAL_PED_BLINK_GO : 7 }
'''

class Marker_traffic_signal:
    def __init__(self):
        print("INIT!")
        rospy.init_node('detected_signal_markers')
        self.inputTopic = rospy.get_param("~topic")
        self.light_pub = rospy.Publisher(self.inputTopic + "/markers", MarkerArray, queue_size = 30)
        self.sub_light = rospy.Subscriber(self.inputTopic, DetectedLightArray, self.convert_light)
        #self.sub_sign = rospy.Subscriber(self.inputTopic, DetectedLightArray, self.convert_sign)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.delay_text_mark_pub = rospy.Publisher(self.inputTopic + "/delay", MarkerArray, queue_size=30)
        self.type_array = [ 'Unknown', 'Red', 'Yellow', 'Green', 'RED_RIGHT']

        self.lifetime = rospy.Duration(1.0)  #for 1 fps update rate



    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def create_traffic_light_set(self, message, light_msg):
        markers = []

        marker0 = Marker()
        marker0.header.frame_id = message.header.frame_id
        marker0.header.stamp = message.header.stamp
        marker0.ns = self.inputTopic
        marker0.action = Marker.ADD
        marker0.id = light_msg.lightType
        marker0.type = Marker.TEXT_VIEW_FACING
        marker0.scale.z = 1
        marker0.lifetime = rospy.Duration(1.0)
        marker0.color.r = 0
        marker0.color.g = 1
        marker0.color.b = 0
        marker0.color.a = 1.0

        marker0.pose.position.x = light_msg.distance
        marker0.pose.position.y = 0
        marker0.pose.position.z = 6.0
        marker0.pose.orientation.w = 1.0
        #text
        if light_msg.classId >= len(self.type_array):
            text = self.type_array[0]
        else:
            text = self.type_array[light_msg.classId]
        marker0.text = "dis: %.2fm %s" % (light_msg.distance, text)
        markers.append(marker0)

        #stick
        marker1 = Marker()
        marker1.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker1.header.frame_id = message.header.frame_id
        marker1.header.stamp = message.header.stamp
        marker1.ns = self.inputTopic
        marker1.action = Marker.ADD
        marker1.pose.orientation.w = 1.0
        marker1.id = light_msg.lightType + 20
        marker1.type = Marker.CYLINDER
        marker1.scale.x = 0.2
        marker1.scale.y = 0.2
        marker1.scale.z = 5.6

        marker1.pose.position.x = light_msg.distance
        marker1.pose.position.y = -0.1
        marker1.pose.position.z = 2.8

        marker1.color.r = 0.8
        marker1.color.g = 0.8
        marker1.color.b = 0.8
        marker1.color.a = 1

        #plane
        marker2 = Marker()
        marker2.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker2.header.frame_id = message.header.frame_id
        marker2.header.stamp = message.header.stamp
        marker2.ns = self.inputTopic
        marker2.action = Marker.ADD
        marker2.pose.orientation.w = 1.0
        marker2.id = light_msg.lightType + 40
        marker2.type = Marker.CUBE
        marker2.scale.x = 0.1
        marker2.scale.y = 1.2
        marker2.scale.z = 0.4

        marker2.pose.position.x = light_msg.distance
        marker2.pose.position.y = -0.8
        marker2.pose.position.z = 5.4


        marker2.color.r = 0.7
        marker2.color.g = 0.7
        marker2.color.b = 0.7
        marker2.color.a = 1

        #blub
        marker3 = Marker()
        marker3.lifetime = self.lifetime  #lifetime depend on 1 / FPS
        marker3.header.frame_id = message.header.frame_id
        marker3.header.stamp = message.header.stamp
        marker3.ns = self.inputTopic
        marker3.action = Marker.ADD
        marker3.id = light_msg.lightType + 60
        marker3.type = Marker.CYLINDER
        marker3.scale.x = 0.4
        marker3.scale.y = 0.4
        marker3.scale.z = 0.1
        marker3.pose.orientation.x = 0.0
        marker3.pose.orientation.y = 1.0
        marker3.pose.orientation.z = 0.0
        marker3.pose.orientation.w = 1.0

        marker3.color.r = 0
        marker3.color.g = 0
        marker3.color.b = 0
        marker3.color.a = 1

        marker3.pose.position.x = light_msg.distance - 0.05
        marker3.pose.position.z = 5.4

        if light_msg.classId is 1 or light_msg.classId is 4:
            marker3.color.r = 1
            marker3.pose.position.y = -1.2
            markers.append(marker3)
        elif light_msg.classId is 2:
            marker3.color.r = 1
            marker3.color.g = 1
            marker3.pose.position.y = -0.8
            markers.append(marker3)
        elif light_msg.classId is 3:
            marker3.color.g = 1
            marker3.pose.position.y = -0.4
            markers.append(marker3)

        markers.append(marker1)
        markers.append(marker2)
    
        if light_msg.classId is 4:
            
            #arrow
            marker4 = Marker()
            marker4.lifetime = self.lifetime  #lifetime depend on 1 / FPS
            marker4.header.frame_id = message.header.frame_id
            marker4.header.stamp = message.header.stamp
            marker4.ns = self.inputTopic
            marker4.action = Marker.ADD
            marker4.pose.orientation.w = 1.0
            marker4.id = light_msg.lightType + 80
            marker4.type = Marker.ARROW

            marker4.scale.x = 0.25
            marker4.scale.y = 0.4
            marker4.scale.z = 0.2

            st = Point()
            st.x = light_msg.distance - 0.05
            st.y = -1.0
            st.z = 5.0
            marker4.points.append(st)
            ed = Point()
            ed.x = light_msg.distance - 0.05
            ed.y = -1.4
            ed.z = 5.0
            marker4.points.append(ed)
            marker4.id = light_msg.classId + 1000
            marker4.color.r = 0
            marker4.color.g = 1
            marker4.color.b = 0
            marker4.color.a = 1
            markers.append(marker4)

            #plane
            marker5 = Marker()
            marker5.lifetime = self.lifetime  #lifetime depend on 1 / FPS
            marker5.header.frame_id = message.header.frame_id
            marker5.header.stamp = message.header.stamp
            marker5.ns = self.inputTopic
            marker5.action = Marker.ADD
            marker5.pose.orientation.w = 1.0
            marker5.id = light_msg.lightType + 100
            marker5.type = Marker.CUBE
            marker5.scale.x = 0.1
            marker5.scale.y = 0.4
            marker5.scale.z = 0.4

            marker5.pose.position.x = light_msg.distance
            marker5.pose.position.y = -1.2
            marker5.pose.position.z = 5.0

            marker5.color.r = 0.7
            marker5.color.g = 0.7
            marker5.color.b = 0.7
            marker5.color.a = 1
            markers.append(marker5)

        return markers

    def create_delay_text_marker(self, idx, header, dis):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_d"
        marker.action = Marker.ADD
        marker.id = idx + 500 # message.track.id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.scale.z = 1
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1.0
        marker.text = "%.3fms" % ((rospy.get_rostime() - header.stamp).to_sec() * 1000.0)

        marker.pose.position.x = dis
        marker.pose.position.y = 0
        marker.pose.position.z = 7
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker

    def convert_light(self, message):
        #print("CALL ONE!")
        marker_list = MarkerArray()
        text_list = MarkerArray()
        idx = 0

        for light_msg in message.lights:
            marker_list.markers.extend(self.create_traffic_light_set(message, light_msg))
            text_list.markers.append( self.create_delay_text_marker( idx, message.header, light_msg.distance))
            idx += 1
        self.light_pub.publish(marker_list)
        self.delay_text_mark_pub.publish(text_list)


    def run(self):
        print("RUN!")
        rospy.spin()


if __name__ == '__main__':
    node = Marker_traffic_signal()
    node.run()

