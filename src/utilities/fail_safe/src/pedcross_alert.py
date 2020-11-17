import time
import heapq
import rospy
from msgs.msg import DetectedObjectArray
from status_level import OK, WARN, UNKNOWN


class PedCrossAlert(object):
    TOPIC = "/PedCross/Alert"
    def __init__(self):
        # internal variables:
        self.fps_low = 1.0
        self.heap = []
        self.sampling_period_in_seconds = 30 / self.fps_low
        self.pedcrossing = False

        # runtime status
        self.status = UNKNOWN
        self.status_str = ""

        rospy.Subscriber(PedCrossAlert.TOPIC, DetectedObjectArray, self._cb)

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
        self.pedcrossing = False;
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
        for obj in msg.objects:
            # Crossing pedestrian is indicated in TrackInfo.is_ready_prediction
            self.pedcrossing = self.pedcrossing or obj.track.is_ready_prediction
