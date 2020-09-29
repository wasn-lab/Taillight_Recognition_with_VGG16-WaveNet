import time
import heapq
import rospy
from msgs.msg import Flag_Info


class CtrlInfo03(object):
    TOPIC = "/Flag_Info03"
    def __init__(self):
        # expected module stats
        self.fps_low = 1
        self.fps_high = 10

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / self.fps_low
        self.aeb_enable = False
        self.acc_enable = False
        self.xbywire_enable = False

        # runtime status
        self.status = "UNKNOWN"
        self.status_str = ""

        rospy.Subscriber(CtrlInfo03.TOPIC, Flag_Info, self._cb)

    def _get_aeb_status(self):
        if self.aeb_enable:
            status = "OK"
            status_str = ""
        else:
            status = "FATAL"
            status_str = "AEB not enabled!"
        return {"module": "AEB",
                "status": status,
                "status_str": status_str}

    def _get_acc_status(self):
        if self.acc_enable:
            status = "OK"
            status_str = ""
        else:
            status = "FATAL"
            status_str = "ACC not enabled!"
        return {"module": "ACC",
                "status": status,
                "status_str": status_str}

    def _get_xbywire_status(self):
        if self.xbywire_enable:
            status = "OK"
            status_str = ""
        else:
            status = "FATAL"
            status_str = "XByWire not enabled!"
        return {"module": "XByWire",
                "status": status,
                "status_str": status_str}

    def get_status_in_list(self):
        ret = [self._get_acc_status(),
               self._get_aeb_status(),
               self._get_xbywire_status()]
        self._reset()
        return ret

    def _reset(self):
        fps = self._get_fps()
        if fps == 0:
            self.aeb_enable = False
            self.acc_enable = False
            self.xbywire_enable = False

    def _get_fps(self):
        return len(self.heap) / self.sampling_period_in_seconds

    def _update_heap(self):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)
        heapq.heappush(self.heap, now)

    def _cb(self, msg):
        self._update_heap()
        self.xbywire_enable = bool(int(msg.Dspace_Flag06))
        self.aeb_enable = bool(int(msg.Dspace_Flag07))
        self.acc_enable = bool(int(msg.Dspace_Flag08))
