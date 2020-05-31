#!/usr/bin/env python2

import copy
import rospy
from std_msgs.msg import (
    Bool,
    String
)
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Point
from msgs.msg import DetectedObjectArray
from rosgraph_msgs.msg import Clock
#
import numpy as np
import fps_calculator as FPS
import signal_analyzer as SA
# Costmap listener
import costmap_listener_ITRI as CLN

BOX_ORDER = [
    0, 1,
    1, 2,
    2, 3,
    3, 0,

    4, 5,
    5, 6,
    6, 7,
    7, 4,

    0, 4,
    1, 5,
    2, 6,
    3, 7
]



class Node:

    def __init__(self):
        rospy.init_node("detected_object_markers")
        self.inputTopic = rospy.get_param("~topic")
        self.c_red = rospy.get_param("~red")
        self.c_green = rospy.get_param("~green")
        self.c_blue = rospy.get_param("~blue")
        self.delay_prefix = rospy.get_param("~delay_prefix", "")
        self.delay_pos_x = rospy.get_param("~delay_pos_x", 3.0)
        self.delay_pos_y = rospy.get_param("~delay_pos_y", 30.0)
        self.is_ignoring_empty_obj = rospy.get_param("~is_ignoring_empty_obj", True)
        self.is_tracking_mode = rospy.get_param("~is_tracking_mode", False)
        self.txt_frame_id = rospy.get_param("~txt_frame_id", "txt_frame")
        self.is_using_costmap_listener = rospy.get_param("~is_using_costmap_listener", True)
        self.t_clock = rospy.Time()
        # FPS
        self.fps_cal = FPS.FPS()
        # Costmap listener
        if self.is_using_costmap_listener:
            param_dict = dict()
            param_dict['costmap_topic_name'] = "/occupancy_grid_wayarea" # "occupancy_grid" #
            param_dict['is_using_ITRI_origin'] = True
            self.costmap_listener = CLN.COSTMAP_LISTENER(param_dict)
            print("[%s] Using costmap listener!" % self.delay_prefix)
        else:
            self.costmap_listener = None
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
        self.box_mark_pub = rospy.Publisher(self.inputTopic + "/bbox", MarkerArray, queue_size=1)
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
        # # FPS
        # signal_name = "absFPS"
        # param_dict = dict()
        # param_dict["low_threshold"] = {"threshold":5.0}
        # self.checker_abs_fps = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # # Latency (500ms)
        # signal_name = "absLatency"
        # param_dict = dict()
        # param_dict["high_avg_threshold"] = {"threshold":0.5}
        # self.checker_abs_latency = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # # Timeout (700ms)
        # signal_name = "timeout"
        # param_dict = dict()
        # param_dict["timeout"] = {"threshold":0.7}
        # self.checker_timeout = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # prob, closest object
        self.checker_nearProb_couting_range = 30.0 # Object in the range will be counted
        self.checker_nearProb_y_range = 10.0 # The valid range of y value of objects
        signal_name = "nearProb"
        param_dict = dict()
        param_dict["low_avg_threshold"] = {"threshold":0.75} # 0.65 0.7 0.8
        self.checker_nearProb = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )
        # prob, average
        signal_name = "avgProb"
        param_dict = dict()
        param_dict["low_avg_threshold"] = {"threshold":0.65}
        self.checker_avgProb = SA.SIGNAL_ANALYZER(module_name=self.delay_prefix, signal_name=signal_name,event_publisher=self.checker_event_pub, param_dict=param_dict )

    def get_confidence_scores(self, objects):
        """
        bbox
        Input:
            objects (list)
        Output:
            (avg_prob, d_min_prob)
        """
        try:
            d_range = self.checker_nearProb_couting_range # float("inf")
        except:
            d_range = float("inf")
        d_min = d_range # float("inf")
        d_min_idx = None
        d_min_prob = 1.0
        #
        sum_prob = 0.0
        obj_count = 0
        for i, _obj in enumerate(objects):
            _prob = _obj.camInfo.prob
            if _prob == 0.0:
                continue
            # Sum
            #-----------------#
            obj_count += 1
            sum_prob += _prob
            #-----------------#
            depth = self._calculate_distance_bbox( _obj.bPoint )
            # Check with map
            #-----------------------#
            is_valid = self.check_bPoint_in_wayarea(  _obj.bPoint )
            if not is_valid:
                continue
            #-----------------------#
            if _obj.bPoint.p0.x > 0.0 and abs(_obj.bPoint.p0.y) < self.checker_nearProb_y_range:
                # Frontal object and the object is not empty
                if (depth < d_min):
                    # Update
                    d_min = depth
                    d_min_idx = i
                    d_min_prob = _prob
        # Post-process
        #--------------------------------#
        if obj_count == 0:
            avg_prob = 1.0
        else:
            avg_prob = sum_prob/obj_count
        #
        return (avg_prob, d_min_prob, d_min)

    def check_bPoint_in_wayarea(self, bPoint):
        is_valid = True
        if self.costmap_listener is not None:
            is_occ = self.costmap_listener.is_occupied_at_point2D( (bPoint.p0.x, bPoint.p0.y))
            is_valid = (not is_occ) if (is_occ is not None) else False
        return is_valid

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
        p.x = self.delay_pos_x
        p.y = self.delay_pos_y
        p.z = 2.0
        return p

    def _calculate_depth_bbox(self, bbox):
        """
        The depth of a bbox is simply the x value of p0.
        """
        return abs(bbox.p0.x)

    def _calculate_distance_bbox(self, bbox):
        """
        The distance of a bbox is the Euclidean distance between origin and (p0+p4)/2.
        """
        point_1 = np.array( (bbox.p0.x, bbox.p0.y) )
        point_2 = np.array( (bbox.p4.x, bbox.p4.y) )
        return (0.5 * np.linalg.norm( (point_1 + point_2) ) )

    def detection_callback(self, message):
        current_stamp = rospy.get_rostime()
        current_latency = (current_stamp - message.header.stamp).to_sec() # sec.
        self.fps_cal.step()
        # print("fps = %f" % self.fps_cal.fps)

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

        # Checkers
        #-------------------------------------------#
        # self.checker_abs_fps.update(self.fps_cal.fps)
        # self.checker_abs_latency.update(current_latency)
        # self.checker_timeout.update()
        #
        avg_prob, d_min_prob, d_min = self.get_confidence_scores(_objects)
        # if d_min_prob > 0.0 and d_min_prob < 1.0:
        #     print("avg_prob = %f, d_min_prob = %f, d_min = %f" % (avg_prob, d_min_prob, d_min))
        self.checker_nearProb.update(d_min_prob)
        self.checker_avgProb.update(avg_prob)
        #-------------------------------------------#



        box_list = MarkerArray()
        delay_list = MarkerArray()
        box_list.markers.append(self.create_bounding_box_list_marker(1, message.header, _objects ) )
        delay_list.markers.append( self.create_delay_text_marker( 1, message.header, current_stamp, self.text_marker_position_origin(), self.fps_cal.fps, _num_removed_obj ) )
        # idx = 1
        # for i in range(len(_objects)):
        #     # point = self.text_marker_position(_objects[i].bPoint)
        #     box_list.markers.append( self.create_bounding_box_marker( idx, message.header, _objects[i].bPoint) )
        #     # delay_list.markers.append( self.create_delay_text_marker( idx, message.header, point) )
        #     idx += 1
        idx = 2
        if self.is_tracking_mode:
            if self.is_showing_track_id:
                for i in range(len(_objects)):
                    obj_id = _objects[i].track.id
                    box_list.markers.append( self.create_tracking_text_marker( idx, message.header, _objects[i].bPoint, obj_id) )
                    idx += 1
        else:
            if self.is_showing_depth:
                for i in range(len(_objects)):
                    # Decide the source of id
                    obj_id = _objects[i].track.id if self.is_showing_track_id else i
                    prob_ = _objects[i].camInfo.prob if _objects[i].camInfo.prob > 0.0 else None
                    box_list.markers.append( self.create_depth_text_marker( idx, message.header, _objects[i].bPoint, obj_id, prob=prob_ ) )
                    idx += 1
        #
        self.box_mark_pub.publish(box_list)
        self.delay_txt_mark_pub.publish(delay_list)


    # def create_bounding_box_marker(self, idx, header, bbox):
    #     marker = Marker()
    #     marker.header.frame_id = header.frame_id
    #     marker.header.stamp = header.stamp
    #     marker.ns = self.inputTopic
    #     marker.action = Marker.ADD
    #     marker.pose.orientation.w = 1.0
    #     marker.id = idx
    #     marker.type = Marker.LINE_LIST
    #     marker.scale.x = 0.2
    #     marker.lifetime = rospy.Duration(1.0)
    #     marker.color.r = self.c_red
    #     marker.color.g = self.c_green
    #     marker.color.b = self.c_blue
    #     marker.color.a = 1.0
    #
    #     point_list = [
    #         bbox.p0,
    #         bbox.p1,
    #         bbox.p2,
    #         bbox.p3,
    #         bbox.p4,
    #         bbox.p5,
    #         bbox.p6,
    #         bbox.p7
    #     ]
    #
    #     for index in BOX_ORDER:
    #         point = point_list[index]
    #         point_msg = Point()
    #         point_msg.x = point.x
    #         point_msg.y = point.y
    #         point_msg.z = point.z
    #         marker.points.append(point_msg)
    #
    #     return marker

    def create_bounding_box_list_marker(self, idx, header, objects):
        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = header.stamp
        marker.ns = self.inputTopic
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.2
        marker.lifetime = rospy.Duration(1.0)
        marker.color.r = self.c_red
        marker.color.g = self.c_green
        marker.color.b = self.c_blue
        marker.color.a = 1.0


        for _i in range(len(objects)):
            bbox = objects[_i].bPoint
            point_list = [
                bbox.p0,
                bbox.p1,
                bbox.p2,
                bbox.p3,
                bbox.p4,
                bbox.p5,
                bbox.p6,
                bbox.p7
            ]

            for index in BOX_ORDER:
                point = point_list[index]
                point_msg = Point()
                point_msg.x = point.x
                point_msg.y = point.y
                point_msg.z = point.z
                marker.points.append(point_msg)

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
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_delay"), scale=2.0 )

    def create_depth_text_marker(self, idx, header, bbox, bbox_id=None, prob=None):
        """
        Generate a text marker for showing depth/distance of object
        """
        point = self.text_marker_position( bbox )
        # depth = self._calculate_depth_bbox( bbox )
        depth = self._calculate_distance_bbox( bbox )
        # Generate text
        if bbox_id is None:
            text = "D=%.2fm" % ( depth )
        else:
            text = "[%d]D=%.2fm" % (bbox_id, depth )
        if prob is not None:
            text += ",P=%.2f" % prob
        scale = 2.0
        return self.text_marker_prototype(idx, header, text, point=point, ns=(self.inputTopic + "_depth"), scale=scale )

    def create_tracking_text_marker(self, idx, header, bbox, bbox_id=None):
        """
        Generate a text marker for showing tracking info.
        """
        point = self.text_marker_position( bbox )
        # Generate text
        text = "<%s>" % str(bbox_id )
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
