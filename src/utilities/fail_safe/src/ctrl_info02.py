# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import time
import heapq
import rospy
from std_msgs.msg import Bool
from msgs.msg import Flag_Info


class CtrlInfo02(object):
    TOPIC = "/Flag_Info02"
    ADV_OP_TOPIC = "/ADV_op/run_state"

    def __init__(self):
        # expected module stats
        self.fps_low = 1
        self.fps_high = 10
        self.msg = None

        # internal variables:
        self.heap = []
        self.sampling_period_in_seconds = 30 / self.fps_low
        rospy.Subscriber(CtrlInfo02.TOPIC, Flag_Info, self._cb)
        self.adv_op_publisher = rospy.Publisher(CtrlInfo02.ADV_OP_TOPIC, Bool, queue_size=1)

    def is_self_driving(self):
        if self.msg is None:
            return False
        return True if int(self.msg.Dspace_Flag08) == 1 else False

    def _reset(self):
        fps = self._get_fps()
        if fps == 0:
            self.msg = None

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
        out_msg = Bool()
        out_msg.data = self.is_self_driving()
        self.adv_op_publisher.publish(out_msg)
