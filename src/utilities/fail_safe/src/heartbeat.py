# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import time
import heapq
import rospy
from object_ids import (OBJECT_ID_PERSON, OBJECT_ID_BICYCLE, OBJECT_ID_MOTOBIKE,
                        OBJECT_ID_CAR)
from message_utils import get_message_type_by_str
from status_level import OK, WARN, ERROR, FATAL, UNKNOWN, OFF, ALARM, NORMAL
from redzone_def import in_3d_roi
from timestamp_utils import get_timestamp_mot


# Global variable:
G_OCC_FAILURE_COUNT = 0

def localization_state_func(_msg, fps):
    """localization stability depends on /current_pose"""
    if fps <= 12:
        # localization drifts when /current_pose fps <= 12
        return FATAL, "FPS: {}".format(fps)
    elif fps < 15:
        # FPS in [12, 15) is tolerable.
        return WARN, "FPS: {}".format(fps)
    return OK, "FPS: {}".format(fps)


def backend_connection_state_func(msg, _fps):
    status = ERROR
    status_str = "No backend connection message"
    if msg is not None:
        connected = msg.data
        if not connected:
            status = ERROR
            status_str = "Cannot connect to backend"
        else:
            status = OK
            status_str = ""

    return status, status_str


def backend_info_func(msg, fps):
    status = ERROR
    status_str = "No backend info message"
    if msg is not None:
        gross_voltage = msg.gross_voltage
        lowest_voltage = msg.lowest_volage
        status = OK
        status_str = "FPS: " + str(fps)[:5]
        if gross_voltage < 350 or lowest_voltage < 3.2:
            status = ERROR
            status_str = ("Battery too low: gross voltage is {}, "
                          "lowest voltage is {}").format(
                              gross_voltage, lowest_voltage)
        elif gross_voltage < 355 or lowest_voltage < 3.25:
            status = WARN
            status_str = ("Low battery: gross voltage is {}, "
                          "lowest voltage is {}").format(
                              gross_voltage, lowest_voltage)
    if status != OK:
        rospy.logwarn("BackendInfo: %s", status_str)
    return status, status_str


def backend_sender_func(msg, fps):
    status = WARN
    status_str = "No message from /backend_sender/status"
    if msg is not None:
        if msg.data:
            status = OK
            status_str = "FPS: " + str(fps)[:5]
        else:
            status = WARN
            status_str = "Cannot send data to backend. FPS: " + str(fps)[:5]

    return status, status_str


def occ_sender_func(msg, fps):
    status = WARN
    status_str = "No message from /occ_sender/status"
    if msg is not None:
        if msg.data:
            status = OK
            status_str = "FPS: " + str(fps)[:5]
            G_OCC_FAILURE_COUNT = 0
        else if G_OCC_FAILURE_COUNT == 0:
            # Packet may not reach OCC due to network issues.
            # If it happens only once, we do not consider it as an unexpected event.
            status = OK
            status_str = "FPS: " + str(fps)[:5]
            G_OCC_FAILURE_COUNT += 1
        else:
            status = WARN
            status_str = "Cannot send data to OCC. FPS: " + str(fps)[:5]
            G_OCC_FAILURE_COUNT += 1

    return status, status_str


__BPOINT_PIDS = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
def __calc_center_by_3d_bpoint(bpoint):
    # only calculate (x, y) now, as z is not so important
    x, y = 0, 0
    for pid in __BPOINT_PIDS:
        x += bpoint.__getattribute__(pid).x
        y += bpoint.__getattribute__(pid).y
    return (x / 8.0, y / 8.0)


def cam_object_detection_func(msg, fps):
    if fps <= 0:
        return ERROR, "FPS: 0"
    if fps > 0 and fps <= 6.0:
        return WARN, "Low FPS: " + str(fps)[:5]

    status = OK
    weak_detection_str = ""
    if msg is not None:
        status = OK
        status_str = "OK"
        for obj in msg.objects:
            center = __calc_center_by_3d_bpoint(obj.bPoint)
            if not in_3d_roi(center[0], center[1]):
                continue
            prob = max(cam_instance.prob for cam_instance in obj.camInfo)
            if prob < 0.6:
                weak_detection_str = ("Low confidence: classId: {}, prob: {}, "
                                      "center: ({:.2f}, {:.2f})").format(
                                          obj.classId, prob, center[0], center[1])
                rospy.logwarn("CameraDetection: %s", weak_detection_str)
    status_str = "FPS: " + str(fps)[:5]
    if weak_detection_str:
        status_str += ", " + weak_detection_str
    return status, status_str


def lidar_detection_func(msg, fps):
    if fps <= 0:
        return ERROR, "FPS: 0"
    if fps > 0 and fps <= 5:
        return WARN, "Low FPS: " + str(fps)[:5]

    status = OK
    weak_detection_str = ""
    if msg is not None:
        for obj in msg.objects:
            if not obj.camInfo:
                # LidarDetection stores the prob in camInfo for now.
                # If we cannot get the prob, just ignore the message.
                continue
            center = __calc_center_by_3d_bpoint(obj.bPoint)
            if center[0] <= 0:
                # We don't care much about objects behind the car.
                continue
            prob = max(cam_instance.prob for cam_instance in obj.camInfo)
            if ((obj.classId == OBJECT_ID_CAR and prob < 0.41) or
                    (obj.classId == OBJECT_ID_PERSON and prob < 0.31) or
                    (obj.classId == OBJECT_ID_BICYCLE and prob < 0.31) or
                    (obj.classId == OBJECT_ID_MOTOBIKE and prob < 0.31)):
                weak_detection_str = (
                    "Low confidence: classId: {}, prob: {}, "
                    "center: ({:.2f}, {:.2f})").format(
                        obj.classId, prob, center[0], center[1])
                rospy.logwarn("LidarDetection: %s", weak_detection_str)
    status_str = "FPS: " + str(fps)[:5]
    if weak_detection_str:
        status_str += ", " + weak_detection_str
    return status, status_str


class Heartbeat(object):
    def __init__(self, module_name, topic, message_type, fps_low, fps_high,
                 inspect_message_contents, latch, sensor_type=None,
                 sensor_uid=None):
        # expected module stats
        self.module_name = module_name
        self.topic = topic
        self.fps_low = fps_low
        self.fps_high = fps_high
        self.latch = latch
        self.enabled = True
        self.inspect_func = None
        self.message_type = message_type
        self.inspect_message_contents = inspect_message_contents
        self.sensor_type = sensor_type  # one of ["camera", "lidar", "radar"]
        self.sensor_uid = sensor_uid  # Only used when sensor_type is defined.
        if module_name == "localization_state":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = localization_state_func

        if module_name == "backend_connection":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = backend_connection_state_func

        if module_name == "3d_object_detection":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = cam_object_detection_func

        if module_name == "backend_info":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = backend_info_func

        if module_name == "LidarDetection":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = lidar_detection_func

        if module_name == "backend_sender":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = backend_sender_func

        if module_name == "occ_sender":
            rospy.logwarn("%s: register inspection function for message", module_name)
            self.inspect_func = occ_sender_func

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / fps_low
        self.msg = None
        self.got_latched_msg = False

        # runtime status
        self.alive = False
        self.status = UNKNOWN
        self.status_str = ""
        self.subscriber = None

        if not self.latch:
            rospy.logwarn("%s: subscribe %s with type %s",
                          self.module_name, self.topic, message_type)
            self.subscriber = rospy.Subscriber(self.topic,
                get_message_type_by_str(self.message_type), self.heartbeat_cb)
        else:
            rospy.logwarn("%s: subscribe latched %s with type %s",
                          self.module_name, self.topic, message_type)

    def to_dict(self):
        self._update_status()
        ret = {"module": self.module_name,
               "status": self.status,
               "status_str": self.status_str}
        if not self.enabled:
            ret["status"] = OK
            ret["status_str"] = "Disabled"
        return ret

    def get_fps(self):
        fps = len(self.heap) / self.sampling_period_in_seconds
        if fps == 0 and self.subscriber is not None:
            rospy.logwarn("Publisher might be down. Reconnect to get topic %s", self.topic)
            self.subscriber.unregister()
            self.subscriber = rospy.Subscriber(self.topic,
                get_message_type_by_str(self.message_type), self.heartbeat_cb)
        return fps

    def _update_status(self):
        self._update_heap()  # Clear out-of-date timestamps
        if self.inspect_func is not None:
            if self.enabled:
                fps = self.get_fps()
                if fps == 0:
                    self.msg = None
                self.status, self.status_str = self.inspect_func(self.msg, fps)
            else:
                self.status = OK
                self.status_str = "Disabled"
            return
        if self.latch:
            self._update_status_latch()
        else:
            self._update_status_heartbeat()

    def _update_status_latch(self):
        if self.got_latched_msg:
            self.status = OK
            self.status_str = "{}: Got latched message.".format(self.topic)
        else:
            self.status = ERROR
            self.status_str = "{}: No latched message.".format(self.topic)

    def _update_status_heartbeat(self):
        fps = self.get_fps()
        if fps >= self.fps_low and fps <= self.fps_high:
            self.status = OK
            self.status_str = "FPS: " + str(fps)[:5]

        if fps > self.fps_high:
            self.status = WARN
            self.status_str = "FPS too high: " + str(fps)[:5]
        if fps < self.fps_low:
            self.status = WARN
            self.status_str = "FPS too low: " + str(fps)[:5]
        if fps == 0:
            if self.module_name == "nav_path_astar_final":
                self.status = FATAL
                self.status_str = "Cannot update local path."
            else:
                self.status = ERROR
                self.status_str = "No message from {}".format(self.topic)

    def _update_heap(self):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)

    def update_latched_message(self):
        self.got_latched_msg = False
        rospy.Subscriber(self.topic, get_message_type_by_str(self.message_type), self.cb_latch)

    def cb_latch(self, msg):
        if self.inspect_func is not None:
            self.msg = msg
        self.got_latched_msg = True

    def heartbeat_cb(self, msg):
        if self.inspect_message_contents:
            self.msg = msg
        heapq.heappush(self.heap, time.time())
        self._update_heap()

    def get_sensor_status(self):
        if self.sensor_type is None:
            return {}
        status = NORMAL
        fps = self.get_fps()
        if fps == 0:
            status = OFF
        elif fps < self.fps_low:
            status = ALARM

        return {"uid": self.sensor_uid,
                "timestamp": get_timestamp_mot(),
                "source_time": get_timestamp_mot(),
                "status": status}

    def get_ego_speed(self):
        if self.msg is None:
            rospy.logerr("%s: No ego_speed, not receive data yet", self.module_name)
            return float("inf")
        return int(self.msg.ego_speed)

    def set_enabled(self, mode):
        print("set {} enable={}".format(self.module_name, mode))
        self.enabled = mode
