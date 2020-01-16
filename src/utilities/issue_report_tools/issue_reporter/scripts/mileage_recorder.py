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
#
is_manual_brake = False
# Queue
manual_brake_Q = Queue.Queue()
#-------------------#

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
    global is_manual_brake, manual_brake_Q
    # print("Dspace_Flag07 = %f" % data.Dspace_Flag07)

    # 0: no manual brake, 1: manually braked
    is_manual_brake_now = data.Dspace_Flag07 > 0.5
    if is_manual_brake != is_manual_brake_now:
        # State change event
        now = rospy.get_rostime()
        is_manual_brake = is_manual_brake_now
        manual_brake_Q.put( (is_manual_brake, now) )
        if is_manual_brake:
            print("Manually brake!!")
        else:
            print("Release manual brake~")


def main(sys_args):
    """
    """
    global mileage_km, speed_mps_filtered
    global is_manual_brake, manual_brake_Q
    #
    rospy.init_node('mileage_recorder', anonymous=False)

    # Loading parameters
    #---------------------------------------------#
    rospack = rospkg.RosPack()
    _pack_path = rospack.get_path('msg_recorder')
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
    # Publisher
    #--------------------------------------#



    # Loop for user command via stdin
    rate = rospy.Rate(5.0) # Hz
    while not rospy.is_shutdown():
        # Do somthing
        evet_str = "mileage: %.3f km, speed(filter): %.1f km/hr" % (mileage_km, speed_mps_filtered*3.6)
        if is_manual_brake:
            evet_str += ", manually braked"
        print(evet_str)
        #
        if not manual_brake_Q.empty():
            manual_brake_event = manual_brake_Q.get()
            # if manual_brake_event[0]:
            #     print("Manually brake!!")
            # else:
            #     print("Release manual brake~")

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
