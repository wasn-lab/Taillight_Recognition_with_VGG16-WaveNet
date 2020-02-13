#!/usr/bin/env python
# license removed for brevity
import rospy
import time
import threading
import SIGNAL_PROCESSING as SP
#-------------------------#
try:
    import queue as Queue # Python 3.x
except:
    import Queue # Python 2.x
#-------------------------#
from std_msgs.msg import (
    Header,
    String,
    Empty,
    Bool,
    Int32,
)

"""
!! Important !!

TODO:
- Modify the sys_ready check method from a "writing directly to global dict"
  to a sliding window method by using Queue.
  --> To solve the "missing detection" problem.
- Add a container for saving latest status during this window.
  In case that there is no event happend during the next window,
  use theis state (latest state) as current state

"""

# Timeouts
#-------------------------#
timeout_alive = 1.5 # sec.
timeout_thread_alive = None
#-------------------------#

"""
Check items
Important note: Use this list to control te components to be checked
"""
#-------------------------#
check_list = ["node_alive", "REC_is_recording"]
check_list += ["brake_status"]
# check_list += ["backend_connected"]
check_list += ["localization_state"]

"""
The startup_check_list is a subset of check_list.
- The status of components in the startup_check_list will defaultly set to "ERROR".
- For components not listed in the list, the status is set to "OK" by default.
"""
startup_check_list = ["node_alive", "REC_is_recording"]
# The following items will be added for release version
# startup_check_list += ["backend_connected"] # Will be added for release version
# startup_check_list += ["localization_state"]


print("check_list = %s" % str(check_list))
print("startup_check_list = %s" % str(startup_check_list))
#-------------------------#



# Definitions
# Note: smaller is better
#-------------------------#
STATE_DEF_dict = dict()
STATE_DEF_dict["OK"] = 0
STATE_DEF_dict["WARN"] = 1
STATE_DEF_dict["ERROR"] = 2
STATE_DEF_dict["FATAL"] = 3
STATE_DEF_dict["UNKNOWN"] = 4
# Generate the inverse mapping of the state definitions
STATE_DEF_dict_inv = dict()
for key in STATE_DEF_dict:
    STATE_DEF_dict_inv[ STATE_DEF_dict[key] ] = key
#-------------------------#


# Level setting
#-------------------------#
LOGGING_LEVEL = STATE_DEF_dict["WARN"] # Greater or equal to this level will be displayed
SYS_FAIL_LEVEL = STATE_DEF_dict["ERROR"] # Greater or equal to this level means the system is failed
REC_BACKUP_LEVEL = STATE_DEF_dict["WARN"] # Greater or equal to this trigger the backup of recorder
#-------------------------#



# States
#-------------------------#
# ADV running state
advop_run_state = False # Default not running

# Initialize the container for status queues
check_queue = dict()
for key in check_list:
    check_queue[key] = Queue.Queue()
# The container for latest status get from queue
check_latest_status_dict = dict()

# Initialize the container for "latest statistic status"
# Note: Initial status solely determin if the item is checked at startup.
# Important: not to write to this check_dict directly, use queue instead
check_dict = dict()
for key in check_list:
    if key in startup_check_list:
        check_dict[key] = STATE_DEF_dict["ERROR"] # For startup check
    else:
        check_dict[key] = STATE_DEF_dict["OK"] # Not checked at startup

# Conclusion
sys_total_status = STATE_DEF_dict["ERROR"]
# ROS message backup
ros_msg_backup = dict()
#-------------------------#

# ROS publishers
#-------------------------------#
ros_advop_sys_ready_pub = rospy.Publisher('/ADV_op/sys_ready', Bool, queue_size=10)
REC_record_backup_pub = rospy.Publisher('/REC/req_backup', String, queue_size=10)
sys_fail_reson_pub = rospy.Publisher('/ADV_op/sys_fail_reason', String, queue_size=100)
#-------------------------------#

# SIGNAL_PROCESSING
#-------------------------------#
run_state_delay = SP.DELAY_CLOSE(delay_sec=3.0, init_state=False)
#-------------------------------#



# Timeout timer
#-------------------------------#
def _timeout_handle_alive():
    """
    """
    # global var_advop_node_alive
    global STATE_DEF_dict
    # global check_dict
    # check_dict["node_alive"] = STATE_DEF_dict["ERROR"] # Abandoned
    global check_queue
    check_queue["node_alive"].put(STATE_DEF_dict["ERROR"])
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
#-------------------------------#





# Evaluate the True/False from status code
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
#
def evaluate_is_REC_BACKUP(status_level):
    """
    """
    global REC_BACKUP_LEVEL
    return ( status_level >= REC_BACKUP_LEVEL )
#--------------------------------------#

# Get and print the logging string
#--------------------------------------#
def get_fail_string(component_status, component_key=""):
    """
    Input:
        - component_status
        - component_key
    Output:
        - _fail_str
    """
    global STATE_DEF_dict, STATE_DEF_dict_inv
    _fail_str = ""
    if evaluate_is_logging(component_status):
        if evaluate_is_fail(component_status):
            _fail_str += "<%s> check fail." % (component_key )
        else:
            _fail_str += "<%s> OK, but status = %s." % (component_key, STATE_DEF_dict_inv[component_status] )
    return  _fail_str

def status_ros_logging(component_status, _fail_str):
    """
    Input:
        - component_status
        - _fail_str
    """
    global STATE_DEF_dict, STATE_DEF_dict_inv
    if evaluate_is_logging(component_status):
        if evaluate_is_fail(component_status):
            rospy.logerr("[sys_ready] %s" % _fail_str )
        else:
            rospy.logwarn("[sys_ready] %s" % _fail_str )
#--------------------------------------#



# Get status codes from ROS messages
#--------------------------------------#
def code_func_bool(msg):
    """
    Output: status_code, event_str
    """
    global STATE_DEF_dict
    state = msg.data
    if state == True:
        return STATE_DEF_dict["OK"], ""
    elif state == False:
        return STATE_DEF_dict["ERROR"], ""
    else:
        return STATE_DEF_dict["UNKNOWN"], ""

def code_func_localization(msg):
    """
    Output: status_code, event_str
    """
    global STATE_DEF_dict
    state = msg.data
    low_gnss_frequency = (state & 1) > 0
    low_lidar_frequency = (state & 2) > 0
    low_pose_frequency = (state & 4) > 0
    pose_unstable = (state & 8) > 0
    #
    status_code = STATE_DEF_dict["OK"]
    if pose_unstable:
        status_code = max(status_code, STATE_DEF_dict["FATAL"])
    if low_pose_frequency:
        status_code = max(status_code, STATE_DEF_dict["WARN"])
    if low_lidar_frequency:
        status_code = max(status_code, STATE_DEF_dict["WARN"])
    if low_gnss_frequency:
        status_code = max(status_code, STATE_DEF_dict["WARN"])
    return status_code, ""

# Brake
brake_state_dict = dict()
brake_state_dict[0] = "Released"
brake_state_dict[1] = "Auto-braked"
brake_state_dict[2] = "Anchored" # Stop-brake
brake_state_dict[3] = "AEB"
brake_state_dict[4] = "Manual brake"
def code_func_brake(msg):
    """
    Output: status_code, event_str
    """
    global STATE_DEF_dict
    global brake_state_dict
    state = msg.data
    print("state = %d" % state)
    if state >= 3:
        status_code = STATE_DEF_dict["WARN"] # Note: not to stop self-driving
    else:
        status_code = STATE_DEF_dict["OK"]
    event_str = brake_state_dict[state]
    return status_code, event_str
#--------------------------------------#

# ROS callbacks
#--------------------------------------#
def _checker_CB(msg, key, code_func=code_func_bool, is_event_msg=True, is_trigger_REC=True, post_func=None ):
    """
    is_event_msg: True-->event message, False-->state message
    """
    global ros_msg_backup, check_queue
    global advop_run_state
    # global ros_msg_backup, check_dict
    _status, _event_str = code_func(msg)
    # check_dict[key] = _status # Note: key may not in check_dict, this can be an add action.
    # Note: key may not in check_dict, this can be an add action.
    if not key in check_queue:
        check_queue[key] = Queue.Queue()
    check_queue[key].put(_status)

    # EVENT trigger REC backup
    if key in check_list: # If it's not in the check_list, bypass the recorder part
        # It should be checked to trigger recorder and publish event
        if is_event_msg or ros_msg_backup.get(key, None) != msg: # Only status change will viewd as event
            # if advop_run_state:
            if run_state_delay.output(): # Note: delayed close
                # Note: We only trigger record if it's already in self-driving mode and running
                #       The events during idle is not going to be backed-up.
                if is_trigger_REC and evaluate_is_REC_BACKUP(_status):
                    # Trigger recorder with reason
                    _reason = "%s:%s:%s" % (key, STATE_DEF_dict_inv[_status], _event_str )
                    REC_record_backup_pub.publish( _reason )
                    # Write some log
                    rospy.logwarn("[sys_ready] REC backup reason:<%s>" % _reason )
                    # Publish the event message
                    #
            #
        #
    #
    if not post_func is None:
        post_func() # e.g. "node_alive" should set its timeout timer
    ros_msg_backup[key] = msg

def ADV_op_run_state_CB(msg):
    """
    """
    global advop_run_state
    advop_run_state = msg.data
    # Delay close
    run_state_delay.input(advop_run_state)
#--------------------------------------#




def main():
    # global var_advop_node_alive, var_REC_is_recording
    global STATE_DEF_dict, STATE_DEF_dict_inv
    global check_list # The list of components needs to be checked
    global check_dict, check_queue, check_latest_status_dict
    global sys_total_status
    rospy.init_node('ADV_sys_ready_check', anonymous=False)
    print("[sys_ready_check] Node started.")

    # Start timers
    set_timer_alive()

    # ROS subscribers
    # Note: The key for callback function should match the checklist
    #-----------------------------#
    # ADV system running state
    # rospy.Subscriber("ADV_op/run_state", Bool, ADV_op_run_state_CB)
    rospy.Subscriber("ADV_op/run_state/republished", Bool, ADV_op_run_state_CB)
    # all_alive from node_trace
    # Note: The "/all_alive"  topic callback should append a timeout watcher
    rospy.Subscriber("/node_trace/all_alive", Bool, (lambda msg: _checker_CB(msg, "node_alive", is_event_msg=False, post_func=set_timer_alive) ) )
    # The following topic can go without timeout watcher (since they are not periodical messages)

    # REC_is_recording
    rospy.Subscriber("/REC/is_recording", Bool, (lambda msg: _checker_CB(msg, "REC_is_recording", is_event_msg=True, is_trigger_REC=False) ) )
    # backend_connected
    rospy.Subscriber("/backend/connected", Bool, (lambda msg: _checker_CB(msg, "backend_connected", is_event_msg=False, is_trigger_REC=False) ) )

    # The following are events (normally not checked at startup)
    # brake_status
    rospy.Subscriber("/mileage/brake_status", Int32, (lambda msg: _checker_CB(msg, "brake_status", is_event_msg=True, code_func=code_func_brake ) ) )
    # Localization (state published in 40 Hz)
    rospy.Subscriber("/localization_state", Int32, (lambda msg: _checker_CB(msg, "localization_state", is_event_msg=False, code_func=code_func_localization ) ) )
    #-----------------------------#



    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():

        # Ready logic
        _sys_status_now = STATE_DEF_dict["OK"]
        _fail_str_list = list()

        # Check through check_list
        #-----------------------------------------------#
        for check_item in check_queue:
            if check_queue[check_item].empty():
                # No event happened in this window
                latest_status = check_latest_status_dict.get(check_item, None)
                if not latest_status is None:
                    # There were some events happened in the previous window
                    check_dict[check_item] = latest_status
                    check_latest_status_dict[check_item] = None
            else:
                # Some events happened in this window
                num_items = check_queue[check_item].qsize()
                worst_status = None
                latest_status = None
                for idx in range(num_items):
                    try:
                        an_item_status = check_queue[check_item].get(False)
                        if (worst_status is None) or (an_item_status > worst_status):
                            worst_status = an_item_status
                        latest_status = an_item_status
                    except:
                        print("check_queue[%s] is empty" % check_item)
                        break
                if not worst_status is None:
                    check_dict[check_item] = worst_status
                check_latest_status_dict[check_item] = latest_status


        # for check_item in check_dict:
        for check_item in check_list:
            # We consider only the items in the check_list
            # Note: check_list is a subset of check_dict keys
            _fail_str_list.append( get_fail_string(check_dict[check_item], check_item) )
            status_ros_logging(check_dict[check_item], _fail_str_list[-1])
            _sys_status_now = max(_sys_status_now, check_dict[check_item] )
        #-----------------------------------------------#

        # Processing string list
        #----------------------------------#
        # Remove empty strings
        try:
            while True:
                _fail_str_list.remove("")
        except:
            pass
        # Combine strings to a single string
        _fail_str = "\n".join(_fail_str_list)
        # print("_fail_str = \n%s" % _fail_str)
        #----------------------------------#

        # Publish the fail string and print a separator if something happend
        #----------------------------------#
        if len(_fail_str_list) > 0:
            print("---") # For stout
            sys_fail_reson_pub.publish(_fail_str) # ROS message
        #----------------------------------#

        # Check is status changed, for logging
        #----------------------------------#
        if _sys_status_now != sys_total_status:
            if evaluate_is_OK(_sys_status_now):
                rospy.loginfo("[sys_ready] The system is ready.")
            else:
                rospy.logwarn("[sys_ready] The system is not ready.")
        sys_total_status = _sys_status_now
        #----------------------------------#

        # Publish ready
        #----------------------------------#
        ros_advop_sys_ready_pub.publish( evaluate_is_OK(sys_total_status) )
        #----------------------------------#
        try:
            rate.sleep()
        except: # For ros time moved backward
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
