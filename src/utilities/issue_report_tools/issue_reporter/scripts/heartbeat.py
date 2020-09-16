import time
import heapq
import rospy
from message_utils import get_message_type_by_str

def localization_state_func(msg):
    if msg is None:
        return "UNKNOWN", "UNKNOWN"
    state = msg.data
    low_gnss_frequency = (state & 1) > 0
    low_lidar_frequency = (state & 2) > 0
    low_pose_frequency = (state & 4) > 0
    pose_unstable = (state & 8) > 0
    status_strs = []
    status = "OK"

    if low_gnss_frequency:
        status = "WARN"
        status_strs.append("low_gnss_frequency")

    if low_lidar_frequency:
        status = "WARN"
        status_strs.append("low_lidar_frequency")

    if low_pose_frequency:
        status = "WARN"
        status_strs.append("low_pose_frequency")

    if pose_unstable:
        status = "FATAL"
        status_strs.append("pose_unstable")

    return status, " ".join(status_strs)


class Heartbeat(object):
    def __init__(self, module_name, topic, message_type, fps_low, fps_high, inspect_message_contents):
        # expected module stats
        self.module_name = module_name
        self.topic = topic
        self.fps_low = fps_low
        self.fps_high = fps_high
        self.inspect_func = None
        if inspect_message_contents:
            if module_name == "localization_state":
                rospy.logwarn("%s: register inspection function for message", module_name)
                self.inspect_func = localization_state_func

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / fps_low
        self.msg = None

        # runtime status
        self.alive = False
        self.status = "UNKNOWN"
        self.status_str = ""

        rospy.logwarn("%s: subscribe %s with type %s", self.module_name, self.topic, message_type)
        ret = rospy.Subscriber(self.topic, get_message_type_by_str(message_type), self.heartbeat_cb)

    def to_dict(self):
        self._update_status()
        return {"module": self.module_name,
                "status": self.status,
                "status_str": self.status_str}

    def _get_fps(self):
        return len(self.heap) / self.sampling_period_in_seconds

    def _update_status(self):
        if self.inspect_func is not None:
            self.status, self.status_str = self.inspect_func(self.msg)
            return
        fps = self._get_fps()
        if fps >= self.fps_low and fps <= self.fps_high:
            self.status = "OK"
            self.status_str = "FPS: {:.2f}".format(fps)

        if fps > self.fps_high:
            self.status = "WARN"
            self.status_str = "FPS too high: {:.2f}".format(fps)
        if fps < self.fps_low:
            self.status = "WARN"
            self.status_str = "FPS too low: {:.2f}".format(fps)
        if fps == 0:
            self.status = "ERROR"
            self.status_str = "Node {} is offline!".format(self.module_name)


    def _update_heap(self):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)
        heapq.heappush(self.heap, now)

    def heartbeat_cb(self, msg):
        if self.inspect_func is not None:
            self.msg = msg
        self._update_heap()
