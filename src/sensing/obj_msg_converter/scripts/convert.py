#!/usr/bin/env python2

from __future__ import division
import rospy
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Point
from std_msgs.msg import Header
from msgs.msg import *
from autoware_perception_msgs.msg import *
import numpy as np


def my_divide(dividend, divisor):
    remain = dividend % divisor
    floor = dividend // divisor
    return floor, remain


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

    def classid_convert(self, input):
        dict = {
            0: 0,
            1: 6,
            2: 4,
            3: 5,
            4: 1,
            5: 3,
            6: 2,
            7: -1,
            8: -1,
            9: -1
        }
        return dict.get(input)

    def msg_convert(self, in_list):
        out_list = DynamicObjectArray()
        out_list.header = in_list.header

        for in_obj in in_list.objects:
            out_obj = DynamicObject()

            # in_obj.track.id(uint32) to out_obj.id.uuid(uint8[16])
            id_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp_id = in_obj.track.id
            tmp_id, id_list[15] = my_divide(tmp_id, 256)
            tmp_id, id_list[14] = my_divide(tmp_id, 256)
            tmp_id, id_list[13] = my_divide(tmp_id, 256)
            tmp_id, id_list[12] = my_divide(tmp_id, 256)
            out_obj.id.uuid = id_list
            # print('in_obj.track.id = {0}'.format(in_obj.track.id))
            # print('out_obj.id.uuid = {0}'.format(out_obj.id.uuid))

            out_obj.semantic.type = self.classid_convert(in_obj.classId)
            # print('in_obj.classId = {0}; out_obj.semantic.type = {1}'.format(in_obj.classId, out_obj.semantic.type))

            out_obj.semantic.confidence = in_obj.camInfo.prob

            out_obj.state.pose_covariance.pose.position.x = (
                in_obj.bPoint.p0.x + in_obj.bPoint.p7.x) / 2
            out_obj.state.pose_covariance.pose.position.y = (
                in_obj.bPoint.p0.y + in_obj.bPoint.p7.y) / 2
            out_obj.state.pose_covariance.pose.position.z = (
                in_obj.bPoint.p0.z + in_obj.bPoint.p7.z) / 2
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
            if not in_obj.cPoint.lowerAreaPoints:
                for p in in_obj.cPoint.lowerAreaPoints:
                    out_obj.shape.footprint.append(Point32(p.x, p.y, p.z))

            out_list.objects.append(out_obj)

        return out_list


if __name__ == "__main__":
    node = Node()
    node.run()
