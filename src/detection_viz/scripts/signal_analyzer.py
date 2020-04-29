#!/usr/bin/env python2

import rospy
import numpy as np
# import threading
from timeit import default_timer as timer
import time
import json

class SIGNAL_ANALYZER(object):

    def __init__(self, signal_name="signal_analysis", event_publisher=None, param_dict={}):
        """
        param_dict: (key: checker_func_key, value: parameters dict of the checker)
        - "high_threshold"
            - "threshold": (0.0) # Note: value in () is the default value
        - "low_threshold"
            - "threshold": (0.0)

        - "high_avg_threshold"
            - "threshold": (0.0)
        - "low_avg_threshold"
            - "threshold": (0.0)

        - "higher_avg_value"
            - "threshold": (0.0)
        - "lower_avg_value"
            - "threshold": (0.0)

        - "higher_avg_ratio"
            - "threshold": (1.0)
        - "lower_avg_ratio"
            - "threshold": (1.0)
        """
        #
        self.name = signal_name
        self.param_dict = param_dict

        # states
        self.value = None
        self.value_avg = None
        self.stamp_start = timer()
        self.stamp_last = self.stamp_start

        # Parameters
        self.alpha = 2*np.pi*0.1 # 0.1 Hz
        # ROS std_msg/String publisher
        self.event_publisher = event_publisher
        # List of checker_func
        self.checker_func_list = []
        # Setup checkers (Note: this function should be overloaded)
        self.setup_checkers()

    def setup_checkers(self):
        """
        Select the checkers to load
        """

        #----------------------------------------#
        checker_key = "high_threshold"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_high_threshold)
        #
        checker_key = "low_threshold"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_low_threshold)
        #----------------------------------------#
        checker_key = "high_avg_threshold"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_high_avg_threshold)
        #
        checker_key = "low_avg_threshold"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_low_avg_threshold)
        #----------------------------------------#
        checker_key = "higher_avg_value"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_higher_avg_value)
        #
        checker_key = "lower_avg_value"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 0.0
            self.checker_func_list.append(self.check_func_lower_avg_value)
        #----------------------------------------#
        checker_key = "higher_avg_ratio"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 1.0
            self.checker_func_list.append(self.check_func_higher_avg_ratio)
        #
        checker_key = "lower_avg_ratio"
        if checker_key in self.param_dict:
            param_1 = "threshold"
            if not param_1 in self.param_dict[checker_key]:
                self.param_dict[checker_key][param_1] = 1.0
            self.checker_func_list.append(self.check_func_lower_avg_ratio)
        #----------------------------------------#

    #----------------------------------------#
    def check_func_high_threshold(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "high_threshold"
        if value_in > self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)

    def check_func_low_threshold(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "low_threshold"
        if value_in < self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)
    #----------------------------------------#
    def check_func_high_avg_threshold(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "high_avg_threshold"
        if self.value_avg > self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)

    def check_func_low_avg_threshold(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "low_avg_threshold"
        if self.value_avg < self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)
    #----------------------------------------#
    def check_func_higher_avg_value(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "higher_avg_value"
        if (value_in - self.value_avg) > self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)

    def check_func_lower_avg_value(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "lower_avg_value"
        if (value_in - self.value_avg) < self.param_dict[checker_key]["threshold"]:
            print(checker_key)
            self.publish_event("WARN", checker_key)
    #----------------------------------------#
    def check_func_higher_avg_ratio(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "higher_avg_ratio"
        check_H = value_in > (self.value_avg * self.param_dict[checker_key]["threshold"])
        check_L = value_in < (self.value_avg * self.param_dict[checker_key]["threshold"])
        #
        check_ = check_H if self.value_avg >= 0.0 else check_L
        if check_:
            print(checker_key)
            self.publish_event("WARN", checker_key)

    def check_func_lower_avg_ratio(self, value_in):
        """
        This is a checker_func.
        """
        checker_key = "lower_avg_ratio"
        check_H = value_in > (self.value_avg * self.param_dict[checker_key]["threshold"])
        check_L = value_in < (self.value_avg * self.param_dict[checker_key]["threshold"])
        #
        check_ = check_L if self.value_avg >= 0.0 else check_H
        if check_:
            print(checker_key)
            self.publish_event("WARN", checker_key)
    #----------------------------------------#


    # ------------------------------------#
    def update(self, value_in):
        """
        This is a function that need to be call at each iteration.
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



    # Private functions
    # ------------------------------------#
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
    # ------------------------------------#
    # end Private functions




if __name__ == "__main__":
    param_dict = dict()
    param_dict["high_threshold"] = {"threshold":0.6}
    param_dict["low_threshold"] = {"threshold":0.4}

    param_dict["high_avg_threshold"] = {"threshold":0.6}
    param_dict["low_avg_threshold"] = {"threshold":0.4}

    param_dict["higher_avg_value"] = {"threshold":0.1}
    param_dict["lower_avg_value"] = {"threshold":-0.1}

    param_dict["higher_avg_ratio"] = {"threshold":1.1}
    param_dict["lower_avg_ratio"] = {"threshold":0.9}


    sig_analyzer_base = SIGNAL_ANALYZER(param_dict=param_dict)
    while True:
        value =  np.random.rand()
        print("value = %f" % value)
        sig_analyzer_base.update( value )
        print("sig_analyzer_base.value_avg = %f" % sig_analyzer_base.value_avg)
        time.sleep(0.3)
