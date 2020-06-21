#!/usr/bin/env python2

from __future__ import division
import rospy
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Point
from std_msgs.msg import Header
from msgs.msg import *
from msgs2.autoware_perception_msgs.msg.object_recognition import *


class Node:

    def __init__(self):
        rospy.init_node("detectedin_obj._msg_converter")

        # Publishers
        self.track3d_aw_pub = rospy.Publisher(
            "/Tracking3D_aw", DynamicObjectArray, queue_size=1)
        # Subscribers
        self.track3d_sub = rospy.Subscriber(
            "/Tracking3D", DetectedObjectArray, self.track3d_callback)

    def run(self):
        rospy.spin()

    def track3d_callback(self, msg):
        self.track3d_aw_pub.publish(self.msg_convert(msg))

    def msg_convert(self, in_list):
        out_list = DynamicObjectArray()

        for in_obj in in_list.objects:
            out_obj = DynamicObject()

            out_obj.id = in_obj.track.id

            out_obj.semantic.type = in_obj.classId
            out_obj.semantic.confidence = in_obj.camInfo.prob

            out_obj.state.pose_covariance.pose.position.x = (
                in_obj.bPoint.b0.x + in_obj.bPoint.b7.x) / 2
            out_obj.state.pose_covariance.pose.position.y = (
                in_obj.bPoint.b0.y + in_obj.bPoint.b7.y) / 2
            out_obj.state.pose_covariance.pose.position.z = (
                in_obj.bPoint.b0.z + in_obj.bPoint.b7.z) / 2
            # out_obj.state.pose_covariance.pose.orientation =
            # out_obj.state.pose_covariance.covariance =

            out_obj.state.twist_covariance.twist.linear.x = in_obj.track.absolute_velocity.x
            out_obj.state.twist_covariance.twist.linear.y = in_obj.track.absolute_velocity.y
            out_obj.state.twist_covariance.twist.linear.z = in_obj.track.absolute_velocity.z
            # out_obj.state.twist_covariance.twist.angular =
            # out_obj.state.twist_covariance.covariance =

            # out_obj.state.accel_covariance.accel.linear =
            # out_obj.state.accel_covariance.accel.angular =
            # out_obj.state.accel_covariance.covariance =

            out_obj.shape.type = 2
            # out_obj.shape.dimensions =
            if not obj.cPoint.lowerAreaPoints.empty():
                for p in cPoint.lowerAreaPoints:
                    out_obj.shape.footprint.append(Point32(p.x, p.y, p.z))

            out_list.objects.append(out_obj)

        return out_list


if __name__ == "__main__":
    node = Node()
    node.run()
