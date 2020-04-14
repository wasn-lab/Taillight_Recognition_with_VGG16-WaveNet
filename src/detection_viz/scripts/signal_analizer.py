#!/usr/bin/env python2

import rospy
import numpy as np
# import threading
from timeit import default_timer as timer
import time
import json

class SIGNAL_ANALYZER(object):

    def __init__(self, signal_name="signal_analysis", event_publisher=None):
        #
        self.name = signal_name

        # states
        self.value = None
        self.value_avg = None
        self.stamp_start = timer()
        self.stamp_last = self.stamp_start

        # Parameters
        self.alpha = 2*np.pi*0.1 # 0.1 Hz

        self.event_publisher = event_publisher

        #
        self.checker_func_list = []
        self.checker_func_list.append(self.sample_check_func)
        # self.event_str_dict = dict()

    def _filter(self, value_in):
        """
        This is a adaptive low pass filter for obtaining the average of the signal.
        """
        stamp_now = timer()
        delta_T = stamp_now - self.stamp_last
        #---------------------------#
        # Update the parameter of LPF
        adT = self.alpha * delta_T
        if adT > 1.0:
            adT = 1.0
        elif adT < 0.0:
            adT = 0.0
        # Update
        #---------------------------#
        self.value_avg += adT * (value_in - self.value_avg)
        self.value = value_in
        self.stamp_last = stamp_now


    def _event_2_json(self, status, event_str):
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
        json_dict["status"] = status
        json_dict["event_str"] = event_str
        return json.dumps(json_dict)

    def publish_event(self, status, event_str):
        """
        This is the publisher for event.
        """
        event_json = self._event_2_json(status, event_str)
        print(event_json)
        if self.event_publisher:
            self.event_publisher.publish( event_json )

    #
    def update(self, value_in):
        """
        """
        # Initialization
        if self.value is None:
            self.value = value_in
        if self.value_avg is None:
            self.value_avg = value_in
        # check_func
        for _check_func in self.checker_func_list:
            _check_func(value_in)
        # Update stored value
        #--------------------#
        self._filter(value_in)

    def sample_check_func(self, value_in):
        """
        """
        delta_value = value_in - self.value_avg
        if delta_value > 0.0:
            print("Hey")
            self.publish_event("WARN", "abnormally high")

class SIGNAL_ANALYZER_TEST(SIGNAL_ANALYZER):
    pass


if __name__ == "__main__":
    # sig_analyzer = SIGNAL_ANALYZER()
    sig_analyzer = SIGNAL_ANALYZER_TEST()
    while True:
        value =  np.random.rand()
        print("value = %f" % value)
        sig_analyzer.update( value )
        print("value_avg = %f" % sig_analyzer.value_avg)
        time.sleep(0.3)
