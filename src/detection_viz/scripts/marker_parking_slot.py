#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point

from msgs.msg import ParkingSlotResult

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from enum import Enum

from rosgraph_msgs.msg import Clock


class Marker_Parking_Slot:
    def __init__(self):
        rospy.init_node('detected_parking_slot_markers')
        self.inputTopic = self.inputTopic = rospy.get_param("~topic")
        self.ps_pub = rospy.Publisher(self.inputTopic + "/markers", MarkerArray, queue_size = 30)
        self.ps_sub = rospy.Subscriber("/parking_slot_result", ParkingSlotResult, self.convert_ps)
        self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)

        self.lifetime = rospy.Duration(1.0)  #for 1 fps update rate


    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def create_pslot_marker(self, pslot):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.lifetime = rospy.Duration(0.1)
        marker.color.r = 0
        marker.color.g = 0
        marker.color.b = 1
        marker.color.a = 1.0
        mps = pslot.marking_points
        marker.points = [Point() for _ in range(5)]
        for i in range(5):
            mp_index = i % 4;
            marker.points[i].x = mps[mp_index].x
            marker.points[i].y = mps[mp_index].y
            marker.points[i].z = mps[mp_index].z

        # marker.pose.position.x = 2
        # marker.pose.position.y = 3.0
        # marker.pose.position.z = -1
        # marker.pose.orientation.w = 1.0
        return marker

    def convert_ps(self, message):
        marker_list = MarkerArray()

        for pslot in message.parking_slots:
            marker_list.markers.append(self.create_pslot_marker(pslot))
        self.ps_pub.publish(marker_list)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = Marker_Parking_Slot()
    node.run()

