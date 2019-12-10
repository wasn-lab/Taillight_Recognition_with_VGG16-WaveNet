#!/usr/bin/env python2

import rospy
import numpy as np
import threading
from timeit import default_timer as timer
import time

class FPS:

    def __init__(self):
        self.period_update = 0.2 # sec.
        self.window_size = 25 # samples
        self.max_cumulated_step = 100000
        #
        self.buffer_cumulated_stamp = [0 for _ in range(self.window_size)]
        self.idx_now = self.correct_index(0)
        #
        self.cumulated_step = self.correct_step(0)
        #
        self.period_window = self.period_update * self.window_size
        self.fps = 0.0

        #------------------------#
        self.stamp_start = timer()
        self.stamp_last = self.stamp_start
        _t = threading.Timer(self.period_update, self._worker)
        _t.daemon = True
        _t.start()

    def step(self):
        """
        """
        self.cumulated_step = self.correct_step(self.cumulated_step + 1)

    def correct_index(self, index_in):
        """
        """
        return int(index_in % self.window_size)

    def correct_step(self, step_in):
        """
        """
        return int(step_in % self.max_cumulated_step)

    def _worker(self):
        stamp_now = timer()
        _t = threading.Timer(self.period_update, self._worker)
        _t.daemon = True
        _t.start()
        # Pop out old one
        step_pre = self.buffer_cumulated_stamp[self.idx_now]
        # Push new one
        self.buffer_cumulated_stamp[self.idx_now] = self.cumulated_step
        # Calculate the difference
        _step_diff = self.buffer_cumulated_stamp[self.idx_now] - step_pre
        _step_diff = self.correct_step(_step_diff)
        # Calculate FPS
        self.fps = _step_diff / float(self.period_window)
        # next step
        self.idx_now = self.correct_index(self.idx_now + 1)

        # print("fps = %f" % self.fps)


        _delta_T = stamp_now - self.stamp_last
        self.stamp_last = stamp_now
        # print("Hey %s sec." % str(_delta_T))




if __name__ == "__main__":
    fps = FPS()
    while True:
        fps.step()
        time.sleep(0.3)
