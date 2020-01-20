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

# States
#-------------------------#
check_dict = dict()
check_dict["node_alive"] = False
check_dict["REC_is_recording"] = False
# var_advop_node_alive = False
# var_REC_is_recording = False
#
var_advop_sys_ready = False
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
    global check_dict
    check_dict["node_alive"] = False
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
def _node_alive_CB(mag):
    """
    """
    # global var_advop_node_alive
    global check_dict
    check_dict["node_alive"] = mag.data
    # var_advop_node_alive = mag.data
    set_timer_alive()

def _REC_is_recording_CB(mag):
    """
    """
    # global var_REC_is_recording
    global check_dict
    check_dict["REC_is_recording"] = mag.data
    # var_REC_is_recording = mag.data
#--------------------------------------#


def get_fail_string(is_component_good, component_name=""):
    """
    Input:
        - is_component_good
        - component_name
    Output:
        - _fail_str
    """
    _fail_str = ""
    if not is_component_good:
        _fail_str += "<%s> check fail.\n" % component_name
        rospy.logerr("[sys_ready] %s" % _fail_str )
    return  _fail_str



def main():
    # global var_advop_node_alive, var_REC_is_recording
    global check_dict
    global var_advop_sys_ready
    rospy.init_node('ADV_sys_ready_check', anonymous=False)
    print("[sys_ready_check] Node started.")

    # Start timers
    set_timer_alive()

    # ROS subscribers
    #-----------------------------#
    rospy.Subscriber("/node_trace/all_alive", Bool, _node_alive_CB)
    rospy.Subscriber("/REC/is_recording", Bool, _REC_is_recording_CB)
    #-----------------------------#



    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():
        # Ready logic
        _sys_ready_now = True
        _fail_str = ""
        # Check list
        #-----------------------------------------------#
        # _fail_str += get_fail_string(var_advop_node_alive, "node_alive")
        # _fail_str += get_fail_string(var_REC_is_recording, "REC_is_recording")
        # _sys_ready_now &= var_advop_node_alive
        # _sys_ready_now &= var_REC_is_recording

        for check_item in check_dict:
            _fail_str += get_fail_string(check_dict[check_item], check_item)
            _sys_ready_now &= check_dict[check_item]
        #-----------------------------------------------#
        # print("_fail_str = \n%s" % _fail_str)
        sys_fail_reson_pub.publish(_fail_str)
        # Changing check
        if _sys_ready_now != var_advop_sys_ready:
            if _sys_ready_now:
                rospy.loginfo("[sys_ready] The system is ready.")
            else:
                rospy.logwarn("[sys_ready] The system is not ready.")
        var_advop_sys_ready = _sys_ready_now
        # Publish ready
        ros_advop_sys_ready_pub.publish(var_advop_sys_ready)
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
