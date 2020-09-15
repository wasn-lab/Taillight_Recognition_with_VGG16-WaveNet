import time
import heapq
import rospy

class Heartbeat():
    def __init__(self, module_name, topic, fps_low, fps_high):
        # expected module stats
        self.module_name = module_name
        self.topic = topic
        self.fps_low = fps_low
        self.fps_high = fps_high

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / fps_low

        # runtime status
        self.alive = False
        self.status = "UNKNOWN"
        self.status_str = ""

    def to_dict(self):
        return {"module": self.module_name,
                "status": self.status,
                "status_str": self.status_str}

    def _get_fps(self):
        return len(self.heap) / self.sampling_period_in_seconds

    def update_status(self):
        self.status = "UNKNOWN"
        self.status_str = "No unexpected events"

        fps = self._get_fps()
        if fps > self.fps_high:
            self.status = "WARN"
            self.status_str = "FPS too high: {}".format(fps)
        if fps < self.fps_low:
            self.status = "WARN"
            self.status_str = "FPS too low: {}".format(fps)
        if fps == 0:
            self.status = "ERROR"
            self.status_str = "Node {} crashed".format(self.module_name)


    def heartbeat_cb(self, msg):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)
        heapq.heappush(self.heap, now)
        rospy.logwarn("%s: fps: %f", self.module_name, self._get_fps())
