#!/usr/bin/env python
import random
import string
import json
import threading
import time
#

# MQTT
#---------------------------#
import paho.mqtt.client as mqtt
#---------------------------#

# ROS
#---------------------------#
import rospy
from std_msgs.msg import (
    Header,
    Float32,
    String,
    Empty,
    Bool,
)
# msgs
from msgs.msg import Flag_Info
#---------------------------#

# Timeouts
#-------------------------#
timeout_sys_ready = 2.0 # sec.
timeout_thread_sys_ready = None
#-------------------------#

# States
#-------------------------#
var_advop_run_state = False
var_advop_sys_ready = False
#-------------------------#


# ROS Publishers
ros_advop_sync_pub = rospy.Publisher('/ADV_op/sync', Empty, queue_size=10)
ros_advop_req_run_stop_pub = rospy.Publisher('/ADV_op/req_run_stop', Bool, queue_size=10)



# MQTT
#------------------------------------------------------------#
mqtt_client = None
mqtt_broker = "127.0.0.1" # localhost
# MQTT topics
mqtt_msg_topic_namespace = 'ADV_op'

# Subscribed topics (MQTT --> ROS)
#------------------------#
mqtt_advop_sync_subT = mqtt_msg_topic_namespace + '/sync'
mqtt_advop_req_run_stop_subT = mqtt_msg_topic_namespace + '/req_run_stop'


# The subscription list
# element: (topic, QoS)
mqtt_subscription_list = list()
# NOTE: since that we are subscrbing to local broker, the loading is tiny for QoS=2
mqtt_subscription_list.append( (mqtt_advop_sync_subT, 2) )
mqtt_subscription_list.append( (mqtt_advop_req_run_stop_subT, 2) )
#------------------------#

# Published topics (MQTT --> dock)
#------------------------#
mqtt_advop_run_state_pubT = mqtt_msg_topic_namespace + '/run_state' #
mqtt_advop_sys_ready_pubT = mqtt_msg_topic_namespace + '/sys_ready' #
#------------------------#

#------------------------------------------------------------#
# end MQTT

# Utilities
#--------------------------------------------------#
def _timeout_handle_sys_ready(is_logging=True):
    """
    """
    global var_advop_sys_ready
    var_advop_sys_ready = False
    mqtt_publish_all_states()
    if is_logging: rospy.logwarn("[ADV_op_gateway] Timeout: sys_ready was not received within %.1f sec." % float(timeout_sys_ready))
    print("[ADV_op_gateway] sys_ready = %s" % str(var_advop_sys_ready))

def set_timer_sys_ready():
    global timeout_thread_sys_ready, timeout_sys_ready
    if not timeout_thread_sys_ready is None:
        timeout_thread_sys_ready.cancel()
    timeout_thread_sys_ready = threading.Timer(timeout_sys_ready, _timeout_handle_sys_ready)
    timeout_thread_sys_ready.start()
#--------------------------------------------------#

# MQTT --> ROS
#------------------------------------------------------#
def mqtt_bool_to_char(bool_in):
    """
    """
    return ("1" if bool_in else "0")

def mqtt_char_to_bool(char_in):
    """
    """
    # return (char_in == "1")
    # Note: Unicode and bytes (ASII) in Python3 can not campare to each other
    # char_in is known to be bytes (ASII) but the following do some general trick for both Python2/3, b"" or u"
    # Simply Compare char_in with both versions of string
    return ( (char_in == b"1") or (char_in == u"1") )
    # try:
    #     return ( char_in.decode() == u"1" ) # try converting to unicode
    # except:
    #     return ( char_in == u"1" ) # It's already unicode

def mqtt_publish_all_states():
    """
    """
    global var_advop_run_state, var_advop_sys_ready
    if mqtt_client is None:
        return
    mqtt_client.publish(mqtt_advop_run_state_pubT, payload=mqtt_bool_to_char(var_advop_run_state), qos=2, retain=False)
    mqtt_client.publish(mqtt_advop_sys_ready_pubT, payload=mqtt_bool_to_char(var_advop_sys_ready), qos=2, retain=False)


# The callback for when the client receives a CONNACK response from the server.
def mqtt_on_connect(client, userdata, flags, rc):
    """
    This is the callback function at connecting the MQTT client.
    """
    rospy.loginfo("MQTT client connected with result code <" + str(rc) + ">.")

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe( mqtt_subscription_list )


# The callback for when a PUBLISH message is received from the server.
'''
def mqtt_on_message(client, userdata, mqtt_msg):
    """
    This is the callback function at MQTT messages (un-matched topics) arrival.
    ** On receiving this message, bypass to ROS.
    """
    print(mqtt_msg.topic + " " + str(mqtt_msg.payload))
'''

def mqtt_advop_sync_CB(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving sync signal.
    ** On receiving this message, bypass to ROS.
    """
    print( '[MQTT]<%s> payload = "%s"' % (mqtt_msg.topic, mqtt_msg.payload) )
    ros_advop_sync_pub.publish( Empty() )
    mqtt_publish_all_states()

def mqtt_advop_req_run_stop_CB(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving req_run_stop signal.
    ** On receiving this message, bypass to ROS.
    """
    print( '[MQTT]<%s> payload = "%s"' % (mqtt_msg.topic, mqtt_msg.payload) )
    req_run_stop = mqtt_char_to_bool(mqtt_msg.payload)
    ros_advop_req_run_stop_pub.publish( req_run_stop ) # "1" or "0"
    print("[MQTT] Request to go" if req_run_stop else "[MQTT] Request to stop")
#------------------------------------------------------#
# end MQTT --> ROS




# ROS callbacks
#------------------------------------------------------------#
def ros_advop_run_state_CB(msg):
    """
    This is the callback function for run_state.
    ** On receiving this message, bypass to MQTT interface.
    """
    global var_advop_run_state
    var_advop_run_state = msg.data # '1'
    print("[ADV_op_gateway] run_state = %s" % str(var_advop_run_state))
    if mqtt_client is None:
        return
    # Publish
    mqtt_client.publish(mqtt_advop_run_state_pubT, payload=mqtt_bool_to_char(msg.data), qos=2, retain=False)

def ros_Flag_02_CB(msg):
    """
    This is the callback function for run_state.
    ** On receiving this message, bypass to MQTT interface.
    """
    global var_advop_run_state
    _run_state = (msg.Dspace_Flag08 > 0.5) # 1 for running, 0 for stopped
    var_advop_run_state = _run_state
    # print("var_advop_run_state = %s" % str(var_advop_run_state))
    rospy.loginfo_throttle(1, "[ADV_op_gateway] run_state = %s" % str(var_advop_run_state))
    if mqtt_client is None:
        return
    # Publish
    mqtt_client.publish(mqtt_advop_run_state_pubT, payload=mqtt_bool_to_char(_run_state), qos=2, retain=False)

def ros_advop_sys_ready_CB(msg):
    """
    This is the callback function for sys_ready.
    ** On receiving this message, bypass to MQTT interface.
    """
    global var_advop_sys_ready
    var_advop_sys_ready = msg.data # '1'
    print("[ADV_op_gateway] sys_ready = %s" % str(var_advop_sys_ready))
    if mqtt_client is None:
        return
    # Publish
    mqtt_client.publish(mqtt_advop_sys_ready_pubT, payload=mqtt_bool_to_char(msg.data), qos=2, retain=False)
    #
    set_timer_sys_ready()
#------------------------------------------------------------#
# end ROS callbacks





def main():
    #
    global mqtt_client
    # ROS
    rospy.init_node('ADV_op_gateway', anonymous=False)

    # Start timers
    set_timer_sys_ready()

    # ROS subscribers
    #-----------------------------#
    # Chose only one of them to receive the run state
    rospy.Subscriber("ADV_op/run_state", Bool, ros_advop_run_state_CB)
    rospy.Subscriber("Flag_Info02", Flag_Info, ros_Flag_02_CB)
    #
    rospy.Subscriber("ADV_op/sys_ready", Bool, ros_advop_sys_ready_CB)
    #-----------------------------#


    # MQTT
    #-------------------------------------------------------------------#
    mqtt_client = mqtt.Client()
    # Callback functions
    mqtt_client.on_connect = mqtt_on_connect
    # mqtt_client.on_message = mqtt_on_message # We don't need to process the message we don't want

    # Subscriber callbacks
    mqtt_client.message_callback_add(mqtt_advop_sync_subT, mqtt_advop_sync_CB)
    mqtt_client.message_callback_add(mqtt_advop_req_run_stop_subT, mqtt_advop_req_run_stop_CB)

    # Connect
    is_connected = False
    while (not is_connected) and (not rospy.is_shutdown()):
        try:
            mqtt_client.connect(mqtt_broker, 1883, 60)
            is_connected = True
            rospy.loginfo("[MQTT] Connected to broker.")
        except:
            rospy.logwarn("[MQTT] Failed to connect to broker, keep trying.")
            time.sleep(1.0)
    # Start working
    mqtt_client.loop_start() # This start the actual work on another thread.

    # Publish the sync signal
    # Republish states
    mqtt_publish_all_states()
    #-------------------------------------------------------------------#




    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():
        # print("running")
        mqtt_publish_all_states()
        try:
            rate.sleep()
        except:
            # For ros time moved backward
            pass

    rospy.logwarn("[ADV_op_gateway] The ADV_op_gateway is going to close.")
    _timeout_handle_sys_ready(False)
    time.sleep(1.0)
    print("[ADV_op_gateway] Leave main()")

if __name__ == '__main__':
    main()
    print("[ADV_op_gateway] Closed.")
