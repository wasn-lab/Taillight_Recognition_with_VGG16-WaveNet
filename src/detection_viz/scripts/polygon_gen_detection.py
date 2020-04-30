#!/usr/bin/env python2

import copy
import rospy
from std_msgs.msg import (
    Bool,
    String
)
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Polygon, PolygonStamped, Point32, Point
from std_msgs.msg import Header
from msgs.msg import *
from rosgraph_msgs.msg import Clock
#
import numpy as np
import fps_calculator as FPS
import signal_analyzer as SA

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
        self.is_ignoring_empty_obj = rospy.get_param("~is_ignoring_empty_obj", False)
        self.is_tracking_mode = rospy.get_param("~is_tracking_mode", False)
        self.txt_frame_id = rospy.get_param("~txt_frame_id", "txt_frame")
        self.t_clock = rospy.Time()
        # FPS
        self.fps_cal = FPS.FPS()
        # Flags
        self.is_showing_depth = True
        self.is_showing_track_id = self.is_tracking_mode
        self.is_overwrite_txt_frame_id = (len(self.txt_frame_id) != 0)
        # Checkers
        #------------------------#
        self.checker_event_pub = rospy.Publisher("/d_viz/checker_event", String, queue_size=1000)
        self.setup_checkers()
        #------------------------#
        # Publishers
        self.polygon_pub = rospy.Publisher(self.inputTopic + "/poly", MarkerArray, queue_size=1)
        self.delay_txt_mark_pub = rospy.Publisher(self.inputTopic + "/delayTxt", MarkerArray, queue_size=1)
        # self.clock_sub = rospy.Subscriber("/clock", Clock, self.clock_CB)
        self.detection_sub = rospy.Subscriber(self.inputTopic, DetectedObjectArray, self.detection_callback)
        self.is_showing_depth_sub = rospy.Subscriber("/d_viz/req_show_depth", Bool, self.req_show_depth_CB)
        self.is_showing_track_id_sub = rospy.Subscriber("/d_viz/req_show_track_id", Bool, self.req_show_track_id_CB)



    def run(self):
        rospy.spin()

    def clock_CB(self, msg):
        self.t_clock = msg.clock

    def req_show_depth_CB(self, msg):
        self.is_showing_depth = msg.data

    def req_show_track_id_CB(self, msg):
        self.is_showing_track_id = msg.data

    def setup_checkers(self):
        """
        Setup signal_analyzer
        """
        # FPS
        signal_name = "absFPS"
        param_dict = dict()
        param_dict["low_threshold"] = {"threshold":5.0}
        self.checker_abs_fps = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # Latency (500ms)
        signal_name = "absLatency"
        param_dict = dict()
        param_dict["high_avg_threshold"] = {"threshold":0.5}
        self.checker_abs_latency = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # Timeout (700ms)
        signal_name = "timeout"
        param_dict = dict()
        param_dict["timeout"] = {"threshold":0.7}
        self.checker_timeout = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )


    def _increase_point_z(self, pointXYZ_in, high):
        pointXYZ_out = PointXYZ()
        pointXYZ_out.x = pointXYZ_in.x
        pointXYZ_out.y = pointXYZ_in.y
        pointXYZ_out.z = pointXYZ_in.z + high
        return pointXYZ_out

    def get_top_area_point_list(self, cPoint):
        topAreaPoints = [self._increase_point_z(_pointXYZ, cPoint.objectHigh) for _pointXYZ in cPoint.lowerAreaPoints]
        return topAreaPoints

    def text_marker_position(self, cPoint):
        point_1 = cPoint.lowerAreaPoints[0]
        p = Point()
        p.x = point_1.x + 1.0 # + 2.0
        p.y = point_1.y
        p.z = point_1.z
        return p

    def text_marker_position_origin(self):
        p = Point()
        p.x = self.delay_pos_x
        p.y = self.delay_pos_y
        p.z = 2.0
        return p

    def _calculate_depth_polygon(self, cPoint):
        """
        The depth of a cPoint is simply the x value of closest point.
        """
        min_d = float('inf')
        for _i in range(len( cPoint.lowerAreaPoints )):
            if abs(cPoint.lowerAreaPoints[_i].x) < min_d:
                min_d = abs(cPoint.lowerAreaPoints[_i].x)
        return min_d

    def _calculate_distance_polygon(self, cPoint):
        """
        The distance of a cPoint is the Euclidean distance between origin and the closest point.
        """
        min_d = float('inf')
        for _i in range(len( cPoint.lowerAreaPoints )):
            distance = np.linalg.norm( np.array((cPoint.lowerAreaPoints[_i].x, cPoint.lowerAreaPoints[_i].y)) )
            if distance < min_d:
                min_d = distance
        return min_d

    def detection_callback(self, message):
        current_stamp = rospy.get_rostime()
        current_latency = (current_stamp - message.header.stamp).to_sec() # sec.
        self.fps_cal.step()
        # print("fps = %f" % self.fps_cal.fps)

        # Checkers
        #-------------------------------------------#
        self.checker_abs_fps.update(self.fps_cal.fps)
        self.checker_abs_latency.update(current_latency)
        self.checker_timeout.update()
        #-------------------------------------------#

        # Clean-up the objects if its distance < 0.0
        #----------------------------------------------#
        _objects = None
        _num_removed_obj = None
        if self.is_ignoring_empty_obj:
            _objects = [_obj for _obj in message.objects if _obj.distance >= 0.0]
            _num_removed_obj = len(message.objects) - len(_objects)
        else:
            _objects = message.objects
        #----------------------------------------------#

        box_list = MarkerArray()
        delay_list = MarkerArray()

        box_list.markers.append(self.create_polygon_list(message.header, _objects, 1))
        delay_list.markers.append( self.create_delay_text_marker( 1, message.header, current_stamp, self.text_marker_position_origin(), self.fps_cal.fps, _num_removed_obj ) )
        # idx = 1
        # for i in range(len(_objects)):
        #     # point = self.text_marker_position(_objects[i].cPoint)
        #     box_list.markers.append(self.create_polygon(message.header, _objects[i].cPoint, idx))
        #     idx += 1
        idx = 2
        if self.is_tracking_mode:
            if self.is_showing_track_id:
                for i in range(len(_objects)):
                    obj_id = _objects[i].track.id
                    box_list.markers.append( self.create_tracking_text_marker( idx, message.header, _objects[i].cPoint, obj_id) )
                    idx += 1
        else:
            if self.is_showing_depth:
                for i in range(len(_objects)):
                    # Decide the source of id
                    obj_id = _objects[i].track.id if self.is_showing_track_id else i
                    prob_ = _objects[i].camInfo.prob if _objects[i].camInfo.prob > 0.0 else None
                    box_list.markers.append( self.create_depth_text_marker( idx, message.header, _objects[i].cPoint, obj_id, prob=prob_) )
                    idx += 1

        #
        self.polygon_pub.publish(box_list)
        self.delay_txt_mark_pub.publish(delay_list)


    # def create_polygon(self, header, cPoint, idx):
    #     marker = Marker()
    #     marker.header.frame_id = header.frame_id
    #     marker.header.stamp = header.stamp
    #     marker.ns = self.inputTopic
    #     marker.action = Marker.ADD
    #     marker.pose.orientation.w = 1.0
    #
    #     marker.id = idx
    #     marker.type = Marker.LINE_STRIP
    #     marker.scale.x = 0.1
    #     marker.lifetime = rospy.Duration(1.0)
    #     marker.color.r = self.c_red
    #     marker.color.g = self.c_green
    #     marker.color.b = self.c_blue
    #     marker.color.a = 1.0
    #
    #     marker.points = []
    #     if len(cPoint.lowerAreaPoints) > 0:
    #         for i in range(len(cPoint.lowerAreaPoints)):
    #             marker.points.append(cPoint.lowerAreaPoints[i])
    #         marker.points.append(cPoint.lowerAreaPoints[0])
    #
    #     return marker

    def create_polygon_list(self, header, objects, idx):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.05 # 0.1
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0

        marker.points = []
        for _i in range(len(objects)):
            cPoint = objects[_i].cPoint

            num_points = len(cPoint.lowerAreaPoints)
            if num_points > 0:
                objectHigh = cPoint.objectHigh
                # Bottom points
                marker.points += [ cPoint.lowerAreaPoints[(i-1)//2] for i in range( num_points*2 )  ]
                #
                # _point_pre = cPoint.lowerAreaPoints[-1]
                # for i in range(len(cPoint.lowerAreaPoints)):
                #     marker.points.append(_point_pre)
                #     marker.points.append(cPoint.lowerAreaPoints[i])
                #     _point_pre = cPoint.lowerAreaPoints[i]

                # Top points
                topAreaPoints = self.get_top_area_point_list(cPoint)
                marker.points += [ topAreaPoints[(i-1)//2] for i in range( num_points*2 )  ]

                # Edge points
                marker.points += [cPoint.lowerAreaPoints[i//2] if i%2==0 else topAreaPoints[i//2] for i in range( num_points*2 ) ]

        return marker

    def create_delay_text_marker(self, idx, header, current_stamp, point, fps=None, _num_removed_obj=None):
        """
        Generate a text marker for showing latency and FPS.
        """
        # Generate text
        if len(str(self.delay_prefix)) > 0:
            text = "[%s] " % str(self.delay_prefix)
        else:
            text = ""
        text += "%.3fms" % ((current_stamp - header.stamp).to_sec() * 1000.0)
        if not fps is None:
            text += " fps = %.1f" % fps
        if not _num_removed_obj is None:
            text += " -%d objs" % _num_removed_obj
        # Overwrite the frame_id of the text
        header_txt = copy.deepcopy(header)
        if self.is_overwrite_txt_frame_id:
            header_txt.frame_id = self.txt_frame_id
        #
        return self.text_marker_prototype(idx, header_txt, text, point=point, ns=(self.inputTopic + "_d"), scale=2.0 )

    def create_depth_text_marker(self, idx, header, cPoint, cPoint_id=None, prob=None):
        """
        Generate a text marker for showing depth/distance of object
        """
        point = self.text_marker_position( cPoint )
        # depth = self._calculate_depth_polygon( cPoint )
        depth = self._calculate_distance_polygon( cPoint )
        # Generate text
        if cPoint_id is None:
            text = "D=%.2fm" % ( depth )
        else:
            text = "[%d]D=%.2fm" % (cPoint_id, depth )
        if prob is not None:
            text += ",P=%.2f" % prob
        scale = 1.0
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_depth"), scale=scale )

    def create_tracking_text_marker(self, idx, header, cPoint, cPoint_id=None):
        """
        Generate a text marker for showing tracking info.
        """
        point = self.text_marker_position( cPoint )
        # Generate text
        text = "<%s>" % str(cPoint_id )
        scale = 1.0
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_tracking"), scale=scale )

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
        marker.lifetime = rospy.Duration(1.0)
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
