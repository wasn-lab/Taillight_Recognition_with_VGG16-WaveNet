import time
import heapq
import rospy
from std_msgs.msg import Int8MultiArray
from status_level import OK, UNKNOWN, FATAL


class CanChecker(object):
    TOPIC = "/control_checker"
    NORMAL = 0
    DOWN = 1
    def __init__(self):
        # expected module stats
        self.fps_low = 8
        self.fps_high = 20

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / self.fps_low
        self.can_encoded_states = []

        # runtime status
        self.module_name = "CAN"
        self.status = UNKNOWN
        self.status_str = ""

        rospy.Subscriber(CanChecker.TOPIC, Int8MultiArray, self._cb)

    def _get_can_status(self):
        if self.can_encoded_states:
            ndown = sum(self.can_encoded_states)
            if self.can_encoded_states[-1] == CanChecker.NORMAL and ndown == 0:
                status = OK
                status_str = ""
            elif self.can_encoded_states[-1] == CanChecker.NORMAL and ndown > 0:
                status = FATAL
                status_str = "CAN communication is not stable"
            else:
                status = FATAL
                status_str = "CAN communication is down"
        else:
            status = FATAL
            status_str = "No message from {}".format(CanChecker.TOPIC)
        return {"module": self.module_name, "status": status, "status_str": status_str}

    def get_status_in_list(self):
        ret = [self._get_can_status()]
        self._reset()
        return ret

    def _reset(self):
        fps = self._get_fps()
        if fps == 0:
            self.can_encoded_states = []

    def _get_fps(self):
        self._update_heap()
        return len(self.heap) / self.sampling_period_in_seconds

    def _update_heap(self):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)

    def _cb(self, msg):
        self._update_heap()
        heapq.heappush(self.heap, time.time())
        if self.can_encoded_states:
            self.can_encoded_states.pop(0)
        self.can_encoded_states.append(CanChecker.NORMAL if sum(msg.data) == 0 else CanChecker.DOWN)
