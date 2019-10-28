#!/usr/bin/env python

import rospy
from std_msgs.msg import String

from msgs.msg import DetectedLane
from msgs.msg import DetectedLaneArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from enum import Enum

from numpy import arange
from rosgraph_msgs.msg import Clock

class Marker_lane_curve:
    def __init__(self):
        #print("INIT!")
        rospy.init_node('detected_lane_markers')
        self.inputTopic = rospy.get_param("~topic")
        self.lane_pub = rospy.Publisher(self.inputTopic + "/markers", MarkerArray, queue_size = 30)
        self.sub = rospy.Subscriber(self.inputTopic, DetectedLaneArray, self.convert)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.delay_text_mark_pub = rospy.Publisher(self.inputTopic + "/delay", MarkerArray, queue_size=30)
        
        self.lifetime = rospy.Duration( 1 )  #for 1 fps update rate

    def clock_CB(self, msg):
        self.t_clock = msg.clock

    ## return points [ Point() ] : 3D point of smapling points
    def sampleCurve(self, lane_msg, dis):
        #if dis < 0:
        #    dis = 25 # default sampling distance
        dis = 25 # default sampling distance
        points = []
        a = lane_msg.a
        b = lane_msg.b
        c = lane_msg.c
        d = lane_msg.d
        for x in arange(0, dis, 0.1):
            p = Point()
            p.x = x
            p.y = ( a * (x**3) + b * (x**2) + c * x + d )
            p.z = 0
            points.append(p)

        return points

    def create_delay_text_marker(self, idx, header, lanePoints):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic + "_d"
        marker.action = Marker.ADD
        marker.id = idx # message.track.id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.scale.z = 1.2
        marker.lifetime = self.lifetime  #for 5 fps update rate
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1.0
        marker.text = "%.3fms" % ((rospy.get_rostime() - header.stamp).to_sec() * 1000.0)
        

        marker.pose.position.x = 25 - idx * 2 #slightly shfit the text
        marker.pose.position.y = lanePoints[-1].y
        marker.pose.position.z = 0.4
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        return marker

    def convert(self, message):
        #print("CALL ONE!")
        marker_list = MarkerArray()
        text_list = MarkerArray()
        idx = 0 

        stop_distance = message.disStopLine
   
        for lane_msg in message.lanes:     
            marker = Marker()
            marker.header.frame_id = message.header.frame_id
            marker.header.stamp = message.header.stamp
            marker.ns = self.inputTopic
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.id = lane_msg.id
            marker.type = Marker.LINE_STRIP
            marker.scale.x = 0.2
            marker.lifetime = self.lifetime  #for 5 fps update rate

            marker.points = self.sampleCurve(lane_msg, 25.0)
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 1

            if lane_msg.id is 0:   #left adj lane blue
                marker.color.b = 1
            elif lane_msg.id is 1: #left ego lane green
                marker.color.g = 1
            elif lane_msg.id is 2: #right ego lane green
                marker.color.g = 1
            elif lane_msg.id is 3: #right adj lane blue
                marker.color.b = 1 
            elif lane_msg.id is 4: #mid guide line white
                marker.color.r = 1
                marker.color.g = 1
                marker.color.b = 1
                marker.color.a = 1

            idx += 1

            marker_list.markers.append(marker)
            text_list.markers.append( self.create_delay_text_marker( idx, message.header, marker.points))

        ##stopline
        for lane_msg in message.lanes:
            if lane_msg.id is 1:  #find left ego lane

                # -1 means no stop line
                if stop_distance > 0:
                    marker_stop = Marker()
                    marker_stop.header.frame_id = message.header.frame_id
                    marker_stop.header.stamp = message.header.stamp
                    marker_stop.ns = self.inputTopic
                    marker_stop.action = Marker.ADD
                    marker_stop.pose.orientation.w = 1.0
                    marker_stop.id = 5
                    marker_stop.type = Marker.LINE_STRIP
                    marker_stop.scale.x = 0.2
                    marker_stop.lifetime = self.lifetime  #for 5 fps update rate

                    marker_stop.color.r = 1
                    marker_stop.color.g = 1
                    marker_stop.color.b = 0
                    marker_stop.color.a = 1

                    p1 = Point()
                    p1.x = stop_distance
                    p1.y = -3.5
                    p2 = Point()
                    p2.x = stop_distance
                    p2.y = 3.5
                    marker_stop.points = [ p1, p2 ]
                    marker_list.markers.append(marker_stop)

                    #stop line distance
                    marker_stop2 = Marker()
                    marker_stop2.header.frame_id = message.header.frame_id
                    marker_stop2.header.stamp = message.header.stamp
                    marker_stop2.ns = self.inputTopic
                    marker_stop2.action = Marker.ADD
                    marker_stop2.id = 6
                    marker_stop2.type = Marker.TEXT_VIEW_FACING
                    marker_stop2.scale.z = 1.2
                    marker_stop2.lifetime = self.lifetime  #for 5 fps update rate              

                    marker_stop2.pose.position.x = stop_distance
                    marker_stop2.pose.position.y = 0
                    marker_stop2.pose.position.z = 3
                    marker_stop2.pose.orientation.x = 0.0
                    marker_stop2.pose.orientation.y = 0.0
                    marker_stop2.pose.orientation.z = 0.0
                    marker_stop2.pose.orientation.w = 1.0

                    marker_stop2.color.r = 1
                    marker_stop2.color.g = 1
                    marker_stop2.color.b = 0
                    marker_stop2.color.a = 1
                    marker_stop2.text = "dis: %.3fm" % (stop_distance)
                    marker_list.markers.append(marker_stop2)

                    #stop line delay
                    marker_delay = Marker()
                    marker_delay.header.frame_id = message.header.frame_id
                    marker_delay.header.stamp = message.header.stamp
                    marker_delay.ns = self.inputTopic + "_d"
                    marker_delay.action = Marker.ADD
                    marker_delay.id = idx - 10 # message.track.id
                    marker_delay.type = Marker.TEXT_VIEW_FACING
                    marker_delay.scale.z = 1.2
                    marker_delay.lifetime = self.lifetime  #for 5 fps update rate
                    marker_delay.color.r = 0
                    marker_delay.color.g = 1
                    marker_delay.color.b = 0
                    marker_delay.color.a = 1.0
                    #marker.text = "distance: %.3fm" % (lane_msg.disStopLine)
                    #marker.text = "distance: %.3fm\n%.3fms" % (lane_msg.disStopLine, (rospy.get_rostime() - message.header.stamp).to_sec() * 1000.0)
                    marker_delay.text = "%.3fms" % ((rospy.get_rostime() - message.header.stamp).to_sec() * 1000.0)

                    marker_delay.pose.position.x = stop_distance
                    marker_delay.pose.position.y = 0
                    marker_delay.pose.position.z = 5
                    marker_delay.pose.orientation.x = 0.0
                    marker_delay.pose.orientation.y = 0.0
                    marker_delay.pose.orientation.z = 0.0
                    marker_delay.pose.orientation.w = 1.0

                    text_list.markers.append(marker_delay)

        self.lane_pub.publish(marker_list)
        self.delay_text_mark_pub.publish(text_list)


    def run(self):
        #print("RUN!")
        rospy.spin()

if __name__ == '__main__':
    node = Marker_lane_curve()
    node.run()

