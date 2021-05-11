# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import time
import heapq
import rospy
from msgs.msg import Flag_Info
from status_level import OK, WARN, FATAL, UNKNOWN


# Flag05 contents in self-driving mode
class BrakeStatus:
    N_UNPRESSED = 0
    Y_OVER_SPEED = 1  # for over-speed
    Y_ANCHORING = 2  # maintaining static to prevent from sliding
    Y_AEB = 3        # AEB event
    Y_MANUAL_BRAKE = 4  # driver press brake pedal
    NUM_STATUS = 5


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
        self.brake_status = {_: 0 for _ in range(BrakeStatus.NUM_STATUS)}
        self.prev_reported_events = {_: False for _ in range(BrakeStatus.NUM_STATUS)}

        # runtime status
        self.status = UNKNOWN
        self.status_str = ""

        rospy.Subscriber(CtrlInfo03.TOPIC, Flag_Info, self._cb)

    def _get_aeb_status(self):
        # AEB is disabled temporarily in the control team because it is reported
        # many noisy points.
        return {"module": "AEB",
                "status": OK,
                "status_str": "Disabled for many noisy points"}
        if self.aeb_enable:
            status = OK
            status_str = ""
        else:
            status = FATAL
            status_str = "AEB not enabled!"
        return {"module": "AEB",
                "status": status,
                "status_str": status_str}

    def _get_acc_status(self):
        if self.acc_enable:
            status = OK
            status_str = ""
        else:
            status = FATAL
            status_str = "ACC not enabled!"
        return {"module": "ACC",
                "status": status,
                "status_str": status_str}

    def _get_xbywire_status(self):
        if self.xbywire_enable:
            status = OK
            status_str = ""
        else:
            status = FATAL
            status_str = "XByWire not enabled!"
        return {"module": "XByWire",
                "status": status,
                "status_str": status_str}

    def __get_aeb_event(self):
        status = OK
        status_str = ""
        module = "aeb_event"

        if self.brake_status == BrakeStatus.Y_AEB:
            if self.prev_aeb_event:
                rospy.loginfo("AEB event has been reported")
            else:
                # New AEB event
                status = WARN
                status_str = "AEB: Automatic emergency brake!"
                self.prev_aeb_event = True
        else:
            self.prev_aeb_event = False
        if status != OK:
            doc = {"module": module,
                   "status": status,
                   "status_str": status_str}
            return [doc]
        return []

    def __get_disengage_event(self):
        status = OK
        status_str = ""
        module = ""

        if self.brake_status == BrakeStatus.Y_MANUAL_BRAKE:
            if self.prev_disengage_event:
                rospy.loginfo("Disengage event has been reported")
            else:
                status = FATAL
                status_str = "Disengage: Driver manually press brake pedals!"
                module = "disengage_event"
                self.prev_disengage_event = True
        else:
            self.prev_disengage_event = False

        if status:
            doc = {"module": module,
                   "status": status,
                   "status_str": status_str}
            return [doc]
        return []

    def get_events_in_list(self):
        return self.__get_disengage_event() + self.__get_aeb_event()

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
        self._update_heap()
        return len(self.heap) / self.sampling_period_in_seconds

    def _update_heap(self):
        now = time.time()
        bound = now - self.sampling_period_in_seconds
        while self.heap and self.heap[0] < bound:
            heapq.heappop(self.heap)

    def _cb(self, msg):
        rospy.logwarn("cb")
        self._update_heap()
        heapq.heappush(self.heap, time.time())
        self.brake_status = int(msg.Dspace_Flag05)
        self.xbywire_enable = bool(int(msg.Dspace_Flag06))
        self.aeb_enable = bool(int(msg.Dspace_Flag07))
        self.acc_enable = bool(int(msg.Dspace_Flag08))
