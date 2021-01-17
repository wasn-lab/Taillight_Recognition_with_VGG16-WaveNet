#!/usr/bin/env python2

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, Pose
from msgs.msg import *
from autoware_perception_msgs.msg import *


class MsgConvert2:

    frame_id_target_ = "map"
    frame_id_source_ = "base_link"
    tf_buffer_ = tf2_ros.Buffer()

    def __init__(self):
        rospy.init_node("detectedin_obj._msg_converter")

        in_topic_ = "LidarDetection"

        # Subscribers
        self.sub_ = rospy.Subscriber(
            in_topic_, DetectedObjectArray, self.callback)
        # Publishers
        self.pub_ = rospy.Publisher(
            str(in_topic_ + "/aw"), DynamicObjectWithFeatureArray, queue_size=1)

    def run(self):
        rospy.spin()

    def callback(self, msg):
        self.pub_.publish(self.msg_convert(msg))

    def classid_convert(self, idx):
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
        return dict.get(idx)

    def convert(self, p, tf_stamped):
        # TF (lidar-to-map) for object pose
        pose_in_lidar = Pose()
        pose_in_lidar.position.x = p.x
        pose_in_lidar.position.y = p.y
        pose_in_lidar.position.z = p.z
        pose_in_lidar.orientation.x = 0
        pose_in_lidar.orientation.y = 0
        pose_in_lidar.orientation.z = 0
        pose_in_lidar.orientation.w = 1

        pose_in_map = Pose()
        pose_in_map = do_transform_pose(pose_in_lidar, tf_stamped)
        p.x = pose_in_map.position.x
        p.y = pose_in_map.position.y
        p.z = pose_in_map.position.z

    def convert_all_to_map_tf(self, obj):
        if obj.header.frame_id != self.frame_id_target_:
            tf_stamped = TransformStamped()

            try:
               tf_stamped = self.tf_buffer_.lookup_transform(self.frame_id_target_, self.frame_id_source_, rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
               return

            convert(obj.center_point, tf_stamped)
            convert(obj.bPoint.p0, tf_stamped)
            convert(obj.bPoint.p1, tf_stamped)
            convert(obj.bPoint.p2, tf_stamped)
            convert(obj.bPoint.p3, tf_stamped)
            convert(obj.bPoint.p4, tf_stamped)
            convert(obj.bPoint.p5, tf_stamped)
            convert(obj.bPoint.p6, tf_stamped)
            convert(obj.bPoint.p7, tf_stamped)

    def msg_convert(self, in_list):
        out_list = DynamicObjectWithFeatureArray()
        out_list.header = in_list.header

        for in_obj in in_list.objects:
            self.convert_all_to_map_tf(in_obj)

            out_obj = DynamicObject()

            # fill semantic
            out_obj.semantic.type = self.classid_convert(in_obj.classId)
            out_obj.semantic.confidence = 1.0
            # print('in_obj.classId = {0}; out_obj.semantic.type = {1}'.format(in_obj.classId, out_obj.semantic.type))

            # fill state
            out_obj.state.pose_covariance.pose.position.x = in_obj.center_point.x
            out_obj.state.pose_covariance.pose.position.y = in_obj.center_point.y
            out_obj.state.pose_covariance.pose.position.z = in_obj.center_point.z
            out_obj.state.pose_covariance.pose.orientation.x = in_obj.heading.x
            out_obj.state.pose_covariance.pose.orientation.y = in_obj.heading.y
            out_obj.state.pose_covariance.pose.orientation.z = in_obj.heading.z
            out_obj.state.pose_covariance.pose.orientation.w = in_obj.heading.w

            out_obj.state.orientation_reliable = False

            # fill shape
            out_obj.shape.type = 0
            out_obj.shape.dimensions.x = in_obj.dimension.length
            out_obj.shape.dimensions.y = in_obj.dimension.width
            out_obj.shape.dimensions.z = in_obj.dimension.height

            out_obj_with_feature = DynamicObjectWithFeature()
            out_obj_with_feature.object = out_obj
            out_list.feature_objects.append(out_obj_with_feature)

        return out_list


if __name__ == "__main__":
    node = MsgConvert2()
    node.run()
    tf_listener = tf2_ros.TransformListener(tf_buffer_)