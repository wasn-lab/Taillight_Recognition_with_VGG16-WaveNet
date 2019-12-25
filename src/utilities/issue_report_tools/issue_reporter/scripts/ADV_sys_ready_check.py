#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import (
    Header,
    String,
    Empty,
    Bool,
)

# States
#-------------------------#
var_advop_node_alive = False
var_REC_is_recording = False
#
var_advop_sys_ready = False
#-------------------------#

ros_advop_sys_ready_pub = rospy.Publisher('/ADV_op/sys_ready', Bool, queue_size=10)


def _node_alive_CB(mag):
    """
    """
    global var_advop_node_alive
    var_advop_node_alive = mag.data

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
        var_advop_sys_ready = True
        # Check list
        var_advop_sys_ready &= var_advop_node_alive
        var_advop_sys_ready &= var_REC_is_recording
        # Publish ready
        ros_advop_sys_ready_pub.publish(var_advop_sys_ready)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
