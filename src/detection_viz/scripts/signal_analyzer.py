#!/usr/bin/env python2

import rospy
import numpy as np
# import threading
from timeit import default_timer as timer
import time
import threading
import json

class SIGNAL_ANALYZER(object):

    def __init__(self, module_name="module", signal_name="signal", event_publisher=None, param_dict={}):
        """
        param_dict: (key: checker_func_key, value: parameters dict of the checker)
        # Note: value in () is the default value
        - "timeout"
            - "threshold": (1.0) # sec.

        - "high_threshold"
            - "threshold": (0.0)
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
        self.module_name = module_name
        self.signal_name = signal_name
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

        # Initial state
        self.initial_state_period = 15.0 # 2.0 # 15.0 # sec.
        self.is_initial_state = True

        # List of checker_func
        self.checker_func_list = []
        self.checker_prev_state_list = []
        # Setup checkers (Note: this function should be overloaded)
        self.setup_checkers()

        # timeout
        self.timeout_thread = None
        self.reset_timeout_timer(is_first=True)



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

        # Initialize the list of previour states
        self.checker_prev_state_list = ["OK" for _ in self.checker_func_list]
        #

    #----------------------------------------#
    def check_func_high_threshold(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "high_threshold"
        target = self.param_dict[checker_key]["threshold"]
        if value_in > target:
            event_str = "%s(%f>%f)" % (checker_key, value_in, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f<=%f)" % (checker_key, value_in, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)

    def check_func_low_threshold(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "low_threshold"
        target = self.param_dict[checker_key]["threshold"]
        if value_in < target:
            event_str = "%s(%f<%f)" % (checker_key, value_in, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f>=%f)" % (checker_key, value_in, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)
    #----------------------------------------#
    def check_func_high_avg_threshold(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "high_avg_threshold"
        target = self.param_dict[checker_key]["threshold"]
        if self.value_avg > target:
            event_str = "%s(%f>%f)" % (checker_key, self.value_avg, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f<=%f)" % (checker_key, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)

    def check_func_low_avg_threshold(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "low_avg_threshold"
        target = self.param_dict[checker_key]["threshold"]
        if self.value_avg < target:
            event_str = "%s(%f<%f)" % (checker_key, self.value_avg, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f>=%f)" % (checker_key, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)
    #----------------------------------------#
    def check_func_higher_avg_value(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "higher_avg_value"
        target = self.param_dict[checker_key]["threshold"]
        if (value_in - self.value_avg) > target:
            event_str = "%s(%f-%f>%f)" % (checker_key, value_in, self.value_avg, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f-%f<=%f)" % (checker_key, value_in, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)

    def check_func_lower_avg_value(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "lower_avg_value"
        target = self.param_dict[checker_key]["threshold"]
        if (value_in - self.value_avg) < target:
            event_str = "%s(%f-%f<%f)" % (checker_key, value_in, self.value_avg, target)
            # print(event_str)
            status = "WARN"
        else:
            event_str = "%s(%f-%f>=%f)" % (checker_key, value_in, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)
    #----------------------------------------#
    def check_func_higher_avg_ratio(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "higher_avg_ratio"
        target = self.param_dict[checker_key]["threshold"]
        check_H = value_in > (self.value_avg * target)
        check_L = value_in < (self.value_avg * target)
        check_ = check_H if self.value_avg >= 0.0 else check_L
        if check_:
            event_str = "%s(%f/%f>%f)" % (checker_key, value_in, self.value_avg, target)
            # print(checker_key)
            status = "WARN"
        else:
            event_str = "%s(%f/%f<=%f)" % (checker_key, value_in, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)

    def check_func_lower_avg_ratio(self, value_in, prev_state=None):
        """
        This is a checker_func.
        """
        checker_key = "lower_avg_ratio"
        target = self.param_dict[checker_key]["threshold"]
        check_H = value_in > (self.value_avg * target)
        check_L = value_in < (self.value_avg * target)
        check_ = check_L if self.value_avg >= 0.0 else check_H
        if check_:
            event_str = "%s(%f/%f<%f)" % (checker_key, value_in, self.value_avg, target)
            # print(checker_key)
            status = "WARN"
        else:
            event_str = "%s(%f/%f>=%f)" % (checker_key, value_in, self.value_avg, target)
            status = "OK"
        return self.publish_event(status, event_str, prev_state)
    #----------------------------------------#

    # Timeout
    #-------------------------------------#
    def _timeout_handle(self):
        """
        """
        checker_key = "timeout>%fsec" % self.param_dict["timeout"]["threshold"]
        print(checker_key)
        self.publish_event("WARN", checker_key)

    def reset_timeout_timer(self, is_first=False):
        """
        """
        if not "timeout" in self.param_dict:
            return
        #
        timeout_sec = self.param_dict["timeout"]["threshold"]
        if is_first:
            timeout_sec += self.initial_state_period
        # print("timeout_sec = %f" % timeout_sec)
        #
        if not self.timeout_thread is None:
            self.timeout_thread.cancel()
        self.timeout_thread = threading.Timer(timeout_sec, self._timeout_handle)
        self.timeout_thread.start()
    #-------------------------------------#

    # ------------------------------------#
    def restart(self):
        """
        Restart the filter
        """
        self.stamp_start = timer()
        self.is_initial_state = True

    def update(self, value_in=0.0):
        """
        This is a function that need to be call at each iteration.
        """
        self.reset_timeout_timer()
        # Initialization
        if self.value is None:
            self.value = value_in
        if self.value_avg is None:
            self.value_avg = value_in
        # check_func
        if not self.is_initial_state:
            for i, _check_func in enumerate(self.checker_func_list):
                # print("(before)self.checker_prev_state_list[%d] = %s" % (i, str(self.checker_prev_state_list[i])))
                self.checker_prev_state_list[i] = _check_func(value_in, self.checker_prev_state_list[i])
                # print("(after)self.checker_prev_state_list[%d] = %s" % (i, str(self.checker_prev_state_list[i])))
        # Update stored value
        #--------------------#
        self._filter(value_in)

        # State control
        #--------------------------------#
        if self.is_initial_state:
            if (timer() - self.stamp_start) > self.initial_state_period:
                self.is_initial_state = False
                rospy.logwarn("signal_analyzer(%s|%s) started" % (self.module_name, self.signal_name))
        #--------------------------------#

    # Private functions
    # ------------------------------------#
    def _filter(self, value_in):
        """
        This is a adaptive low pass filter for obtaining the average of the signal.
        """
        stamp_now = timer()
        # Initial state
        if self.is_initial_state:
            self.value_avg = value_in
            self.value = value_in
            self.stamp_last = stamp_now
            return

        # else, after initial state
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
        checker_name = self.module_name + "_" + self.signal_name # self.module_name
        json_dict["module"] = checker_name
        json_dict["status"] = status
        json_dict["event_str"] = "[%s]%s" % (self.signal_name, event_str)
        return json.dumps(json_dict)

    def publish_event(self, status, event_str, prev_state=None):
        """
        This is the publisher for event.
        Note: if the prev_state is not given, it always publish te status.
        """
        # print("status:prev_state = %s:%s" % (str(status), str(prev_state)))
        if status != prev_state:
            event_json = self._event_2_json(status, event_str)
            print(event_json)
            if self.event_publisher:
                self.event_publisher.publish( event_json )
        return status
    # ------------------------------------#
    # end Private functions




if __name__ == "__main__":
    param_dict = dict()
    param_dict["timeout"] = {"threshold":0.2}

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
