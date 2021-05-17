# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import time
import heapq
import json
import rospy
from std_msgs.msg import String
from status_level import OK
from load_collector import SYSTEM_LOADS_TOPIC


class SystemLoadChecker(object):
    def __init__(self):
        # expected module stats
        self.fps_low = 0.5
        self.fps_high = 1.2

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 5.0

        # runtime status
        self.status = OK
        self.status_str = ""
        self.msg = None

        rospy.Subscriber(SYSTEM_LOADS_TOPIC, String, self._cb)

    def get_status_in_list(self):
        fps = self._get_fps()
        doc = {"module": "system_loads"}
        status = OK
        status_strs = ["FPS: " + str(fps)[:5]]
        if self.msg is not None:
            jdata = json.loads(self.msg.data)
            for ipc in jdata:
                ipc_load = jdata[ipc]
                ipc_status = ipc_load["status"]
                ipc_status_str = ipc_load.get("status_str", "")
                status = max(status, ipc_status)
                if ipc_status != OK or ipc_status_str:
                    status_strs.append("{} ({})".format(ipc, ipc_status_str))
        doc["status"] = status
        doc["status_str"] = ", ".join(status_strs)
        return [doc]

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
        self.msg = msg
