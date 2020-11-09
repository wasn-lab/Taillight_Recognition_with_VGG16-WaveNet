import time
import heapq
import rospy
from msgs.msg import PedObjectArray
from status_level import OK, WARN, UNKNOWN


class PedCrossAlert(object):
    TOPICS = ["/PedCross/Pedestrians/front_bottom_60",
              "/PedCross/Pedestrians/front_top_far_30",
              "/PedCross/Pedestrians/left_back_60",
              "/PedCross/Pedestrians/right_back_60"]
    def __init__(self):
        # internal variables:
        self.fps_low = 10
        self.heap = []
        self.sampling_period_in_seconds = 30 / self.fps_low
        self.pedcrossing = False

        # runtime status
        self.status = UNKNOWN
        self.status_str = ""

        for topic in PedCrossAlert.TOPICS:
            rospy.Subscriber(topic, PedObjectArray, self._cb)

    def get_events_in_list(self):
        if self.pedcrossing:
            doc = {"module": "pedcross",
                   "status": WARN,
                   "status_str": "pedestrian is crossing"}
            self.pedcrossing = False
            return [doc]
        return []

    def get_status_in_list(self):
        fps = self._get_fps()
        if fps > self.fps_low:
            doc = {"module": "pedcross",
                   "status": OK,
                   "status_str": ""}
        else:
            doc = {"module": "pedcross",
                   "status": WARN,
                   "status_str": "low fps: {}".format(fps)}
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
        for pedobj in msg.objects:
            if pedobj.crossProbability > 0.6:
                self.pedcrossing = True

