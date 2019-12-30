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
var_advop_node_alive = False
var_REC_is_recording = False
#
var_advop_sys_ready = False
#-------------------------#


ros_advop_sys_ready_pub = rospy.Publisher('/ADV_op/sys_ready', Bool, queue_size=10)

#
def _timeout_handle_alive():
    """
    """
    global var_advop_node_alive
    var_advop_node_alive = False
    rospy.logwarn("[sys_ready] Timeout: sys_alive.")

# ROS callbacks
def _node_alive_CB(mag):
    """
    """
    global var_advop_node_alive
    global timeout_thread_alive
    var_advop_node_alive = mag.data
    if not timeout_thread_alive is None:
        timeout_thread_alive.cancel()
    timeout_thread_alive = threading.Timer(timeout_alive, _timeout_handle_alive)
    timeout_thread_alive.start()

def _REC_is_recording_CB(mag):
    """
    """
    global var_REC_is_recording
    var_REC_is_recording = mag.data

def main():
    global var_advop_node_alive, var_REC_is_recording
    global var_advop_sys_ready
    rospy.init_node('ADV_sys_ready_check', anonymous=False)
    #
    rospy.Subscriber("/node_trace/all_alive", Bool, _node_alive_CB)
    rospy.Subscriber("/REC/is_recording", Bool, _REC_is_recording_CB)


    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():
        # Ready logic
        var_advop_sys_ready_now = True
        # Check list
        var_advop_sys_ready_now &= var_advop_node_alive
        var_advop_sys_ready_now &= var_REC_is_recording
        # Changing check
        if var_advop_sys_ready_now != var_advop_sys_ready:
            if var_advop_sys_ready_now:
                rospy.loginfo("[sys_ready] The system is ready.")
            else:
                rospy.logwarn("[sys_ready] The system is not ready.")
        var_advop_sys_ready = var_advop_sys_ready_now
        # Publish ready
        ros_advop_sys_ready_pub.publish(var_advop_sys_ready)
        rate.sleep()
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
