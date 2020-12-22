#!/usr/bin/env python2

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from std_msgs.msg import Header
from msgs.msg import *


class Node:

    def __init__(self):
        rospy.init_node("pp_ellipses")

        self.in_topic_ = rospy.get_param("~in_topic")

        self.sub_pp_ = rospy.Subscriber(
            self.in_topic_, DetectedObjectArray, self.callback_pp)
        self.pub_pp_ = rospy.Publisher(
            self.in_topic_ + "/pp", MarkerArray, queue_size=1)

    def create_pp_ellipse(self, header, obj, idx, forecast_seq):
        marker = Marker()

        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.in_topic_

        marker.id = idx
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.lifetime = rospy.Duration(1.0)

        marker.pose.position.x = obj.track.forecasts[forecast_seq].position.x
        marker.pose.position.y = obj.track.forecasts[forecast_seq].position.y
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        scale = obj.speed_abs * (forecast_seq + 1) / 200.
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = 0.1

        marker.color.r = 0.8
        marker.color.g = 0.9 - forecast_seq * 0.035
        marker.color.b = 0.0
        marker.color.a = 0.3 - forecast_seq * 0.005

        return marker

    def callback_pp(self, message):
        # select objects which is_ready_prediction from message.objects
        objects = None
        objects = [
            obj for obj in message.objects if (
                obj.track.is_ready_prediction)]

        # create and fill pp_ellipse_list
        pp_ellipse_list = MarkerArray()
        idx = 0
        for obj in objects:
            for seq in range(20):
                pp_ellipse_list.markers.append(
                    self.create_pp_ellipse(
                        message.header, obj, idx, seq))
                idx += 1

        self.pub_pp_.publish(pp_ellipse_list)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = Node()
    node.run()
