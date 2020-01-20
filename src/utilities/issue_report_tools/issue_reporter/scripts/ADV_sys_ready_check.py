#!/usr/bin/env python
# license removed for brevity
import rospy
import time
import threading
from std_msgs.msg import (
    Header,
    String,
    Empty,
    Bool,
)

# Timeouts
#-------------------------#
timeout_alive = 1.5 # sec.
timeout_thread_alive = None
#-------------------------#

# Definitions
# Note: smaller is better
#-------------------------#
STATE_DEF_dict = dict()
STATE_DEF_dict["OK"] = 0
STATE_DEF_dict["WARN"] = 1
STATE_DEF_dict["ERROR"] = 2
STATE_DEF_dict["UNKNOWN"] = 3
# Generate the inverse mapping of the state definitions
STATE_DEF_dict_inv = dict()
for key in STATE_DEF_dict:
    STATE_DEF_dict_inv[ STATE_DEF_dict[key] ] = key
#-------------------------#


# Level setting
#-------------------------#
LOGGING_LEVEL = STATE_DEF_dict["WARN"] # Greater or equal to this level will be displayed
SYS_FAIL_LEVEL = STATE_DEF_dict["ERROR"] # Greater or equal to this level means the system is failed
#-------------------------#

# States
#-------------------------#
check_dict = dict()
check_dict["node_alive"] = STATE_DEF_dict["ERROR"]
check_dict["REC_is_recording"] = STATE_DEF_dict["ERROR"]
#
sys_total_status = STATE_DEF_dict["ERROR"]
#-------------------------#

# ROS publishers
#-------------------------------#
ros_advop_sys_ready_pub = rospy.Publisher('/ADV_op/sys_ready', Bool, queue_size=10)
sys_fail_reson_pub = rospy.Publisher('/ADV_op/sys_fail_reason', String, queue_size=100)
#-------------------------------#

#
def _timeout_handle_alive():
    """
    """
    # global var_advop_node_alive
    global STATE_DEF_dict
    global check_dict
    check_dict["node_alive"] = STATE_DEF_dict["ERROR"]
    # var_advop_node_alive = False
    rospy.logwarn("[sys_ready] Timeout: sys_alive was not received within %.1f sec." % float(timeout_alive) )

def set_timer_alive():
    """
    """
    global timeout_thread_alive, timeout_alive
    if not timeout_thread_alive is None:
        timeout_thread_alive.cancel()
    timeout_thread_alive = threading.Timer(timeout_alive, _timeout_handle_alive)
    timeout_thread_alive.start()


# ROS callbacks
#--------------------------------------#
# Check if node is alive
# def _node_alive_CB(msg):
#     """
#     """
#     # global var_advop_node_alive
#     global check_dict
#     check_dict["node_alive"] = msg.data
#     # var_advop_node_alive = msg.data
#     set_timer_alive()
#
# def _REC_is_recording_CB(msg):
#     """
#     """
#     # global var_REC_is_recording
#     global check_dict
#     check_dict["REC_is_recording"] = msg.data
#     # var_REC_is_recording = msg.data

def eva_func_bool(x):
    """
    """
    global STATE_DEF_dict
    if x == True:
        return STATE_DEF_dict["OK"]
    elif x == False:
        return STATE_DEF_dict["ERROR"]
    else:
        return STATE_DEF_dict["UNKNOWN"]


def _checker_CB(msg, key, eva_func=eva_func_bool, post_func=None ):
    """
    """
    global check_dict
    check_dict[key] = eva_func(msg.data)
    if not post_func is None:
        post_func()
#--------------------------------------#




#--------------------------------------#
def evaluate_is_logging(status_level):
    """
    """
    global LOGGING_LEVEL
    return (status_level >= LOGGING_LEVEL)

def evaluate_is_fail(status_level):
    """
    """
    global SYS_FAIL_LEVEL
    return ( status_level >= SYS_FAIL_LEVEL )
#
def evaluate_is_OK(status_level):
    return (not evaluate_is_fail(status_level))
#--------------------------------------#


def get_fail_string(component_status, component_name=""):
    """
    Input:
        - component_status
        - component_name
    Output:
        - _fail_str
    """
    global STATE_DEF_dict, STATE_DEF_dict_inv
    _fail_str = ""
    if evaluate_is_logging(component_status):
        if evaluate_is_fail(component_status):
            _fail_str += "<%s> check fail.\n" % component_name
            rospy.logerr("[sys_ready] %s" % _fail_str )
        else:
            _fail_str += "<%s> OK, but status = %s.\n" % (component_name, STATE_DEF_dict_inv[component_status] )
            rospy.logwarn("[sys_ready] %s" % _fail_str )
    return  _fail_str



def main():
    # global var_advop_node_alive, var_REC_is_recording
    global STATE_DEF_dict, STATE_DEF_dict_inv
    global check_dict
    global sys_total_status
    rospy.init_node('ADV_sys_ready_check', anonymous=False)
    print("[sys_ready_check] Node started.")

    # Start timers
    set_timer_alive()

    # ROS subscribers
    #-----------------------------#
    # Note: The "/all_alive"  topic callback should append a timeout watcher
    rospy.Subscriber("/node_trace/all_alive", Bool, (lambda msg: _checker_CB(msg, "node_alive", post_func=set_timer_alive) ) )
    # The following topic can go without timeout watcher (since they are not periodical messages)
    rospy.Subscriber("/REC/is_recording", Bool, (lambda msg: _checker_CB(msg, "REC_is_recording")))
    #-----------------------------#



    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():
        # Ready logic
        _sys_status_now = STATE_DEF_dict["OK"]
        _fail_str = ""
        # Check list
        #-----------------------------------------------#
        for check_item in check_dict:
            _fail_str += get_fail_string(check_dict[check_item], check_item)
            _sys_status_now = max(_sys_status_now, check_dict[check_item] )
        #-----------------------------------------------#
        # print("_fail_str = \n%s" % _fail_str)
        sys_fail_reson_pub.publish(_fail_str)
        # Changing check
        if _sys_status_now != sys_total_status:
            if evaluate_is_OK(_sys_status_now):
                rospy.loginfo("[sys_ready] The system is ready.")
            else:
                rospy.logwarn("[sys_ready] The system is not ready.")
        sys_total_status = _sys_status_now
        # Publish ready
        ros_advop_sys_ready_pub.publish( evaluate_is_OK(sys_total_status) )
        try:
            rate.sleep()
        except:
            # For ros time moved backward
            pass
    #
    rospy.logwarn("[sys_ready] The sys_ready check is going to close.")
    # ros_advop_sys_ready_pub.publish(False)
    time.sleep(0.5)
    print("[sys_ready] Leave main()")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    print("[sys_ready] Closed.")
