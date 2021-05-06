#!/usr/bin/env python
import rospy, rospkg
import math
import time
import sys, os
import subprocess
import threading
import yaml, json
# File operations
import datetime
# import dircache # <-- Python2.x only, repalce with os.listdir
import shutil
# Args
import argparse
#-------------------------#
try:
    import queue as Queue # Python 3.x
except:
    import Queue # Python 2.x
#-------------------------#
from std_msgs.msg import (
    Empty,
    Bool,
    String,
    Int32,
    Int8MultiArray,
)

from msgs.msg import (
    VehInfo,
    Flag_Info,
)

# Global variables
#-------------------#
mileage_km = 0.0
last_speed_ros_time = None
speed_mps_filtered = 0.0 # m/sec.

# State
adv_run_state = 0
brake_state = 0
# Module states
Xbywire_run_state = None # 0
AEB_run_state = None # 0
ACC_run_state = None # 0
# CAN state
CAN_up_state = None

# Queue
adv_run_Q = Queue.Queue()
brake_Q = Queue.Queue()
#-------------------#

# ROS publisher
#-------------------#
run_state_pub = rospy.Publisher("ADV_op/run_state/republished", Bool, queue_size=10, latch=True)
text_marker_pub = rospy.Publisher("/mileage/status_text", String, queue_size=10, latch=True)
mileage_json_pub = rospy.Publisher("/mileage/relative_mileage", String, queue_size=10, latch=True)
# brake_event_pub = rospy.Publisher('/mileage/brake_status', Int32, queue_size=100, latch=True)
brake_event_pub = rospy.Publisher('/mileage/brake_event', String, queue_size=100, latch=True)
# Module states
Xbywire_run_event_pub = rospy.Publisher('/mileage/Xbywire_run', String, queue_size=100, latch=True)
AEB_run_event_pub = rospy.Publisher('/mileage/AEB_run', String, queue_size=100, latch=True)
ACC_run_event_pub = rospy.Publisher('/mileage/ACC_run', String, queue_size=100, latch=True)
CAN_up_OK_event_pub = rospy.Publisher('/mileage/CAN_up_OK', String, queue_size=100, latch=True)
#-------------------#


# Define the string of state
#--------------------------------#
# adv_run
adv_run_state_dict = dict()
adv_run_state_dict[0] = "Stopped"
adv_run_state_dict[1] = "Self-driving"
# Brake
brake_state_dict = dict()
brake_state_dict[0] = "Released"
brake_state_dict[1] = "Auto-braked"
brake_state_dict[2] = "Anchored" # Stop-brake
brake_state_dict[3] = "AEB"
brake_state_dict[4] = "Manual brake"
#--------------------------------#

#-------------------------------------------#
def state_2_string(state_dict_in, state_in, str_undefined_state="Undefined state"):
    if state_in in state_dict_in:
        return state_dict_in[state_in]
    else:
        return str_undefined_state
#-------------------------------#
def adv_run_state_2_string(state_in):
    """
    """
    global adv_run_state_dict
    return state_2_string(adv_run_state_dict, state_in, \
                str_undefined_state="Undefined adv_run state")

def brake_state_2_string(state_in):
    """
    """
    global brake_state_dict
    return state_2_string(brake_state_dict, state_in, \
                str_undefined_state="Undefined brake state")
#-------------------------------------------#


# Event report: json_str converters
#--------------------------------#
def brake_state_2_json(state_in, checker_name="brake", check_state=(3,4) ):
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
    json_dict["module"] = checker_name
    if state_in in check_state: # 3 and 4
        json_dict["status"] = "WARN"
    else:
        json_dict["status"] = "OK"
    json_dict["event_str"] = brake_state_2_string(state_in)
    return json.dumps(json_dict)

def module_run_state_2_json(state_in, checker_name="module_run"):
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
    json_dict["module"] = checker_name
    if state_in < 1:
        json_dict["status"] = "FATAL"
    else: # 1
        json_dict["status"] = "OK"
    json_dict["event_str"] = "%s-%s" % (checker_name, "Yes" if state_in >=1 else "No" )
    return json.dumps(json_dict)

def CAN_up_OK_state_2_json(state_in, checker_name="CAN_up_OK"):
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
    json_dict["module"] = checker_name
    if state_in > 0:
        json_dict["status"] = "FATAL"
    else: # 0
        json_dict["status"] = "OK"
    json_dict["event_str"] = "%s-%s" % (checker_name, "Yes" if state_in == 0 else "No" )
    return json.dumps(json_dict)
#--------------------------------#


def calculate_mileage(speed_mps):
    """
    This is the engine for calculating mileage.
    """
    global mileage_km, last_speed_ros_time, speed_mps_filtered
    #
    if last_speed_ros_time is None:
        last_speed_ros_time = rospy.get_rostime()
    now = rospy.get_rostime()
    delta_t = (now - last_speed_ros_time).to_sec()
    if delta_t < 0.0:
        delta_t = 0.0
    last_speed_ros_time = now
    speed_mps = int(speed_mps * 100) / 100.0  # round to cm to avoid floating point error
    #
    speed_mps_filtered += 0.1*(speed_mps - speed_mps_filtered)
    #
    mileage_km += (speed_mps_filtered * delta_t)*0.001
    # print("mileage: %f km, speed(filter): %f(%f) km/hr" % (mileage_km, speed_mps*3.6, speed_mps_filtered*3.6) )

#---------------------------------------------#

def _veh_info_CB(data):
    """
    The callback function of vehicle info.
    """
    # print("ego_speed = %f" % data.ego_speed)
    calculate_mileage( data.ego_speed )


def _flag_info_02_CB(data):
    """
    The callback function of vehicle info.
    """
    global adv_run_state, adv_run_Q
    # print("Dspace_Flag07 = %f" % data.Dspace_Flag07)

    # run state
    adv_run_state_now = int( round(data.Dspace_Flag08) )
    if adv_run_state != adv_run_state_now:
        # State change event
        now = rospy.get_rostime()
        adv_run_state = adv_run_state_now
        adv_run_Q.put( (adv_run_state, now) )
        # Print to stdout
        print( adv_run_state_2_string(adv_run_state) )
        # Publish as ROS message
        run_state_pub.publish( (adv_run_state==1) )


def _flag_info_03_CB(data):
    """
    The callback function of vehicle info.
    """
    global brake_state, brake_Q
    global brake_event_pub
    global Xbywire_run_state, AEB_run_state, ACC_run_state
    global Xbywire_run_event_pub, AEB_run_event_pub, ACC_run_event_pub
    # print("Dspace_Flag07 = %f" % data.Dspace_Flag07)

    # brake state
    brake_state_now = int( round(data.Dspace_Flag05) )
    if brake_state != brake_state_now:
        # State change event
        now = rospy.get_rostime()
        brake_state = brake_state_now
        brake_Q.put( (brake_state, now) )
        # Print to stdout
        print( brake_state_2_string(brake_state) )
        # Publish as ROS message
        # brake_event_pub.publish( brake_state_2_json(brake_state) )
        #
        # AEB checker and manual-brake checker
        brake_event_pub.publish( brake_state_2_json(brake_state, checker_name="brake_AEB", check_state=(3,) ) )
        brake_event_pub.publish( brake_state_2_json(brake_state, checker_name="brake_MINT", check_state=(4,) ) )


    # Xbywire states
    Xbywire_run_state_now = int( round(data.Dspace_Flag06) )
    if Xbywire_run_state != Xbywire_run_state_now:
        # State change event
        Xbywire_run_state = Xbywire_run_state_now
        # Print to stdout
        print( "Xbywire_run_state = %d" % Xbywire_run_state )
        # Publish as ROS message
        Xbywire_run_event_pub.publish( module_run_state_2_json(Xbywire_run_state, checker_name="Xbywire_run") )

    # AEB states
    AEB_run_state_now = int( round(data.Dspace_Flag07) )
    if AEB_run_state != AEB_run_state_now:
        # State change event
        AEB_run_state = AEB_run_state_now
        # Print to stdout
        print( "AEB_run_state = %d" % AEB_run_state )
        # Publish as ROS message
        AEB_run_event_pub.publish( module_run_state_2_json(AEB_run_state, checker_name="AEB_run") )

    # ACC states
    ACC_run_state_now = int( round(data.Dspace_Flag08) )
    if ACC_run_state != ACC_run_state_now:
        # State change event
        ACC_run_state = ACC_run_state_now
        # Print to stdout
        print( "ACC_run_state = %d" % ACC_run_state )
        # Publish as ROS message
        ACC_run_event_pub.publish( module_run_state_2_json(ACC_run_state, checker_name="ACC_run") )

def _control_checker_CB(msg):
    """
    The callback function of CAN status
    """
    global CAN_up_state
    flag_check_list = msg.data
    CAN_up_fail_now = False
    for _s in flag_check_list:
        CAN_up_fail_now |= (_s == 1)
    CAN_up_state_now = 1 if CAN_up_fail_now else 0
    if CAN_up_state != CAN_up_state_now:
        # State change event
        CAN_up_state = CAN_up_state_now
        # Print to stdout
        print( "CAN_up_state = %d" % CAN_up_state )
        # Publish as ROS message
        CAN_up_OK_event_pub.publish( CAN_up_OK_state_2_json(CAN_up_state, checker_name="CAN_up_OK") )


#---------------------#
mileage_json_time_period = 2 # sec.
mileage_json_km_period = 1.0 # 1 km
# Variables
mileage_json_last_timestamp = int(time.time())
mileage_json_last_mileage = 0.0
mileage_json_last_is_self_driving = False
def publish_mileage(total_mileage_in, is_self_driving=False, is_MINT=False):
    """
    """
    global mileage_json_pub
    global mileage_json_last_timestamp, mileage_json_last_mileage
    global mileage_json_last_is_self_driving

    now_sec = int(time.time())
    if (now_sec - mileage_json_last_timestamp) < mileage_json_time_period:
        if (total_mileage_in - mileage_json_last_mileage) < mileage_json_km_period:
            if mileage_json_last_is_self_driving == is_self_driving and (not is_MINT):
                # Nothing changed
                return
    # Mileage reach distance period or time reach time period
    # Publish mileage as json string
    #------------------------------------#
    mileage_data_dict = dict()
    mileage_data_dict["t_start"] = mileage_json_last_timestamp # Currently, last time
    mileage_data_dict["t_end"] = now_sec # Now
    mileage_data_dict["delta_km"] = (total_mileage_in - mileage_json_last_mileage)
    mileage_data_dict["drive_mode"] = "A" if mileage_json_last_is_self_driving else "M" # Note: use the past status
    if is_MINT:
        mileage_data_dict["MINT"] = True
    #
    mileage_json_str = json.dumps(mileage_data_dict)
    mileage_json_pub.publish(mileage_json_str)
    #------------------------------------#
    # Update
    mileage_json_last_timestamp = now_sec
    mileage_json_last_mileage = total_mileage_in
    mileage_json_last_is_self_driving = is_self_driving



def main(sys_args):
    """
    """
    global text_marker_pub
    global mileage_km, speed_mps_filtered
    global brake_state, brake_Q
    global adv_run_state
    #
    rospy.init_node('mileage_recorder', anonymous=False)

    """
    # Loading parameters
    #---------------------------------------------#
    rospack = rospkg.RosPack()
    _pack_path = rospack.get_path('issue_reporter')
    print("_pack_path = %s" % _pack_path)
    f_path = _pack_path + "/params/"
    # Param file name
    f_name_params = "mileage_setting.yaml"
    #----------------#
    params_raw = ""
    with open( (f_path+f_name_params),'r') as _f:
        params_raw = _f.read()
    param_dict = yaml.load(params_raw)
    # Print the params
    print("\nSettings (in json format):\n%s" % json.dumps(param_dict, indent=4))
    #---------------------------------------------#

    # Creating directories if necessary
    #---------------------------------------------#
    output_dir_tmp = param_dict["output_dir_tmp"]
    # Add '/' at the end
    if output_dir_tmp[-1] != "/":
        output_dir_tmp += "/"
    # Preprocessing for parameters
    output_dir_tmp = os.path.expandvars( os.path.expanduser(output_dir_tmp) )
    print("output_dir_tmp = %s" % output_dir_tmp)
    # Creating directories
    try:
        _out = subprocess.check_output(["mkdir", "-p", output_dir_tmp], stderr=subprocess.STDOUT)
        print("The directory <%s> has been created." % output_dir_tmp)
    except:
        print("The directry <%s> already exists." % output_dir_tmp)
        pass
    #---------------------------------------------#
    """


    # Init ROS communication interface
    #--------------------------------------#
    # Subscriber
    rospy.Subscriber("/veh_info", VehInfo, _veh_info_CB)
    rospy.Subscriber("/Flag_Info02", Flag_Info, _flag_info_02_CB)
    rospy.Subscriber("/Flag_Info03", Flag_Info, _flag_info_03_CB)
    rospy.Subscriber("/control_checker", Int8MultiArray, _control_checker_CB)
    #--------------------------------------#

    # Publishing intial state
    #--------------------------------------#
    run_state_pub.publish( (adv_run_state==1) )
    # brake_event_pub.publish( brake_state_2_json(brake_state) )
    # AEB checker and manual-brake checker
    brake_event_pub.publish( brake_state_2_json(brake_state, checker_name="brake_AEB", check_state=(3,) ) )
    brake_event_pub.publish( brake_state_2_json(brake_state, checker_name="brake_MINT", check_state=(4,) ) )
    #--------------------------------------#

    # Loop for user command via stdin
    rate = rospy.Rate(5.0) # Hz
    while not rospy.is_shutdown():
        # Logging, publishing information of mileage, speed, run state, and brake status
        evet_str = "mileage: %.3f km, speed(filter): %.1f km/hr" % (mileage_km, speed_mps_filtered*3.6)
        evet_str += "\t| " + adv_run_state_2_string(adv_run_state)
        evet_str += "\t| " + brake_state_2_string(brake_state)
        # print(evet_str)
        rospy.loginfo_throttle(1.0, evet_str)
        # Publish status as String
        text_marker_pub.publish("%.3fkm, %.1fkm/h, R: %s, B: %s" % (mileage_km, speed_mps_filtered*3.6, adv_run_state_2_string(adv_run_state), brake_state_2_string(brake_state) ))
        #
        if not adv_run_Q.empty():
            adv_run_event = adv_run_Q.get()
            # # Print to stdout
            # print( adv_run_state_2_string(adv_run_event[0]) )

        is_MINT = False
        if not brake_Q.empty():
            brake_event = brake_Q.get()
            # # Print to stdout
            # print( brake_state_2_string(brake_event[0]) )
            if brake_event[0] == 4:
                is_MINT = True

        # Publish mileage as json string
        is_self_driving = (adv_run_state == 1)
        publish_mileage(mileage_km, is_self_driving=is_self_driving, is_MINT=is_MINT)
        #
        try:
            rate.sleep()
        except:
            # For ros time moved backward
            pass
    print("End of main loop.")




if __name__ == '__main__':

    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
    print("End of mileage_recorder.")
