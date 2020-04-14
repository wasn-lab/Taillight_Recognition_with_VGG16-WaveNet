#!/usr/bin/env python2

import rospy
import numpy as np
# import threading
from timeit import default_timer as timer
import time
import json

class SIGNAL_ANALYZER:

    def __init__(self, signal_name="signal_analysis", th_abnormally_high=1.0):
        #
        self.name = signal_name

        # states
        self.value = None
        self.value_avg = None
        #
        self.stamp_start = timer()
        self.stamp_last = self.stamp_start

        # Parameters
        self.alpha = 2*np.pi*0.1 # 0.1 Hz
        self.th_abnormally_high = th_abnormally_high
        #
        self.event_str_dict = dict()
        self.event_str_dict[0] = "abnormally high"

        #
        self.event_publisher = None

    def event_2_json(self, event_code):
        """
        Through ROS std_msgs.String
        json string:
        {
            "module": "yyy"
            "status": "OK"/"WARN"/"ERROR"/"FATAL"/"UNKNOWN"
            "event_str": "xxx event of yyy module"
        }
        Output: json string
        """
        json_dict = dict()
        json_dict["module"] = self.name
        json_dict["status"] = "WARN"
        json_dict["event_str"] = self.event_str_dict[event_code]
        return json.dumps(json_dict)

    def update(self, value_in):
        """
        """
        stamp_now = timer()
        delta_T = stamp_now - self.stamp_last
        #---------------------------#
        if self.value is None:
            self.value = value_in
        if self.value_avg is None:
            self.value_avg = value_in
        #
        delta_value_short = value_in - self.value
        delta_value_long = value_in - self.value_avg
        # Check 1
        if delta_value_long > self.th_abnormally_high:
            event_json = self.event_2_json(0)
            print(event_json)
            if self.event_publisher:
                self.event_publisher.publish( event_json )
        #
        adT = self.alpha * delta_T
        if adT > 1.0:
            adT = 1.0
        elif adT < 0.0:
            adT = 0.0
        # Update the average
        self.value_avg += adT * (value_in - self.value_avg)
        #---------------------------#
        self.value = value_in
        self.stamp_last = stamp_now



if __name__ == "__main__":
    sig_analyzer = SIGNAL_ANALYZER()
    while True:
        sig_analyzer.update( np.random.rand() )
        time.sleep(0.3)
