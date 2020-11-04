import time
import heapq
import rospy
from message_utils import get_message_type_by_str
from status_level import OK, WARN, ERROR, FATAL, UNKNOWN, OFF, ALARM, NORMAL
from redzone_def import in_3d_roi

def localization_state_func(msg):
    if msg is None:
        return ERROR, "No localizaton state message"
    state = msg.data
    low_gnss_frequency = (state & 1) > 0
    low_lidar_frequency = (state & 2) > 0
    low_pose_frequency = (state & 4) > 0
    pose_unstable = (state & 8) > 0
    status_strs = []
    status = OK

    if low_gnss_frequency:
        status = WARN
        status_strs.append("low_gnss_frequency")

    if low_lidar_frequency:
        status = WARN
        status_strs.append("low_lidar_frequency")

    if low_pose_frequency:
        status = WARN
        status_strs.append("low_pose_frequency")

    if pose_unstable:
        status = FATAL
        status_strs.append("pose_unstable")

    return status, " ".join(status_strs)


def backend_connection_state_func(msg):
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


def backend_info_func(msg):
    status = ERROR
    status_str = "No backend info message"
    if msg is not None:
        gross_voltage = msg.gross_voltage
        lowest_voltage = msg.lowest_volage
        status = OK
        status_str = ""
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
    return status, status_str


__BPOINT_PIDS = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
def __calc_center_by_3d_bpoint(bpoint):
    # only calculate (x, y) now, as z is not so important
    x, y = 0, 0
    for pid in __BPOINT_PIDS:
        x += bpoint.__getattribute__(pid).x
        y += bpoint.__getattribute__(pid).y
    return (x / 8.0, y / 8.0)


def cam_object_detection_func(msg):
    status = ERROR
    status_str = "No camera 3d detection result"
    if msg is not None:
        status = OK
        status_str = ""
        for obj in msg.objects:
            center = __calc_center_by_3d_bpoint(obj.bPoint)
            if not in_3d_roi(center[0], center[1]):
                continue
            prob = obj.camInfo.prob
            if prob < 0.6:
                status = WARN
                status_str = ("Low confidence: classId: {}, prob: {}, "
                              "center: ({:.2f}, {:.2f})").format(
                                  obj.classId, prob, center[0], center[1])
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

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / fps_low
        self.msg = None
        self.got_latched_msg = False

        # runtime status
        self.alive = False
        self.status = UNKNOWN
        self.status_str = ""

        if not self.latch:
            rospy.logwarn("%s: subscribe %s with type %s",
                          self.module_name, self.topic, message_type)
            rospy.Subscriber(self.topic, get_message_type_by_str(message_type), self.heartbeat_cb)
        else:
            rospy.logwarn("%s: subscribe latched %s with type %s",
                          self.module_name, self.topic, message_type)

    def to_dict(self):
        self._update_status()
        return {"module": self.module_name,
                "status": self.status,
                "status_str": self.status_str}

    def get_fps(self):
        return len(self.heap) / self.sampling_period_in_seconds

    def _update_status(self):
        self._update_heap()  # Clear out-of-date timestamps
        if self.inspect_func is not None:
            if self.get_fps() == 0:
                self.msg = None
            self.status, self.status_str = self.inspect_func(self.msg)
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
            self.status_str = "FPS: {:.2f}".format(fps)

        if fps > self.fps_high:
            self.status = WARN
            self.status_str = "FPS too high: {:.2f}".format(fps)
        if fps < self.fps_low:
            self.status = WARN
            self.status_str = "FPS too low: {:.2f}".format(fps)
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
                "timestamp": int(time.time()),
                "source_time": int(time.time()),
                "status": status}

    def get_ego_speed(self):
        if self.msg is None:
            rospy.logerr("%s: No ego_speed, not receive data yet", self.module_name)
            return float("inf")
        return int(self.msg.ego_speed)
