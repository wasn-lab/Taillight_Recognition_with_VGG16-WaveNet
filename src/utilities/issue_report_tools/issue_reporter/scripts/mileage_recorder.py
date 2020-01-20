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

# Queue
adv_run_Q = Queue.Queue()
brake_Q = Queue.Queue()
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


def _flag_info_03_CB(data):
    """
    The callback function of vehicle info.
    """
    global brake_state, brake_Q
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



def main(sys_args):
    """
    """
    global mileage_km, speed_mps_filtered
    global brake_state, brake_Q
    #
    rospy.init_node('mileage_recorder', anonymous=False)

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



    # Init ROS communication interface
    #--------------------------------------#
    # Subscriber
    rospy.Subscriber("/veh_info", VehInfo, _veh_info_CB)
    rospy.Subscriber("/Flag_Info02", Flag_Info, _flag_info_02_CB)
    rospy.Subscriber("/Flag_Info03", Flag_Info, _flag_info_03_CB)
    # Publisher
    # Publisher
    text_marker_pub = rospy.Publisher("/mileage/status_text", String, queue_size=10, latch=True) #
    #--------------------------------------#



    # Loop for user command via stdin
    rate = rospy.Rate(5.0) # Hz
    while not rospy.is_shutdown():
        # Do somthing
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
            # print( adv_run_state_2_string(adv_run_state) )
        if not brake_Q.empty():
            brake_event = brake_Q.get()
            # # Print to stdout
            # print( brake_state_2_string(brake_state) )
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
