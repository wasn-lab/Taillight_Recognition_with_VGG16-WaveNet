#!/usr/bin/env python
"""
Interaction between IPCs and pad. The latter will sends self-drving commands.
"""
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
    String,
    Empty,
    Bool,
)
# Timeouts
#-------------------------#
TIMEOUT_SYS_READY = 2.0 # sec.
TIMEOUT_THREAD_SYS_READY = None
#-------------------------#

# States
#-------------------------#
VAR_ADVOP_RUN_STATE = False
VAR_ADVOP_SYS_READY = False
#-------------------------#


# ROS Publishers
ROS_ADVOP_SYNC_PUB = rospy.Publisher('/ADV_op/sync', Empty, queue_size=10)
ROS_ADVOP_REQ_RUN_STOP_PUB = rospy.Publisher('/ADV_op/req_run_stop', Bool, queue_size=10)



# MQTT
#------------------------------------------------------------#
MQTT_CLIENT = None
MQTT_BROKER_IP = "127.0.0.1" # localhost
# MQTT topics
MQTT_TOPIC_NAMESPACE = 'ADV_op'

# Subscribed topics (MQTT --> ROS)
#------------------------#
MQTT_ADVOP_SYNC_SUBT = MQTT_TOPIC_NAMESPACE + '/sync'
MQTT_ADVOP_REQ_RUN_STOP_SUBT = MQTT_TOPIC_NAMESPACE + '/req_run_stop'


# The subscription list
# element: (topic, QoS)
MQTT_SUBSCRIPTION_LIST = list()
# NOTE: since that we are subscrbing to local broker, the loading is tiny for QoS=2
MQTT_SUBSCRIPTION_LIST.append((MQTT_ADVOP_SYNC_SUBT, 2))
MQTT_SUBSCRIPTION_LIST.append((MQTT_ADVOP_REQ_RUN_STOP_SUBT, 2))
#------------------------#

# Published topics (MQTT --> dock)
#------------------------#
MQTT_ADVOP_RUN_STATE_PUBT = MQTT_TOPIC_NAMESPACE + '/run_state' #
#mqtt_advop_event_json_pubT = MQTT_TOPIC_NAMESPACE + '/event_json' #
#------------------------#

#------------------------------------------------------------#
# end MQTT

# Utilities
#--------------------------------------------------#
def _timeout_handle_sys_ready(is_logging=True):
    """
    """
    global VAR_ADVOP_SYS_READY  # pylint: disable=global-statement
    VAR_ADVOP_SYS_READY = False
    mqtt_publish_all_states()
    if is_logging:
        rospy.logwarn("Timeout: no sys_ready in %.1f sec.", TIMEOUT_SYS_READY)
    rospy.loginfo("[ADV_op_gateway] sys_ready = %s", str(VAR_ADVOP_SYS_READY))

def set_timer_sys_ready():
    global TIMEOUT_THREAD_SYS_READY  # pylint: disable=global-statement
    if not TIMEOUT_THREAD_SYS_READY is None:
        TIMEOUT_THREAD_SYS_READY.cancel()
    TIMEOUT_THREAD_SYS_READY = threading.Timer(TIMEOUT_SYS_READY, _timeout_handle_sys_ready)
    TIMEOUT_THREAD_SYS_READY.start()
#--------------------------------------------------#

# MQTT --> ROS
#------------------------------------------------------#
def mqtt_bool_to_char(bool_in):
    """
    """
    return "1" if bool_in else "0"

def mqtt_char_to_bool(char_in):
    """
    """
    # return (char_in == "1")
    # Note: Unicode and bytes (ASII) in Python3 can not campare to each other
    # char_in is known to be bytes (ASII) but the following do some general
    # trick for both Python2/3, b"" or u"
    # Simply Compare char_in with both versions of string
    return (char_in == b"1") or (char_in == u"1")
    # try:
    #     return ( char_in.decode() == u"1" ) # try converting to unicode
    # except:
    #     return ( char_in == u"1" ) # It's already unicode

def mqtt_publish_all_states():
    """
    """
    global VAR_ADVOP_RUN_STATE  # pylint: disable=global-statement
    if MQTT_CLIENT is None:
        return
    run_state = mqtt_bool_to_char(VAR_ADVOP_RUN_STATE)
    rospy.logdebug("publish mqtt message - run_state: %s", run_state)
    MQTT_CLIENT.publish(MQTT_ADVOP_RUN_STATE_PUBT, payload=run_state, qos=2, retain=False)


# The callback for when the client receives a CONNACK response from the server.
def mqtt_on_connect(client, _userdata, _flags, _rc):
    """
    This is the callback function at connecting the MQTT client.
    """
    rospy.loginfo("MQTT client connected with result code <" + str(_rc) + ">.")

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_SUBSCRIPTION_LIST)


# The callback for when a PUBLISH message is received from the server.
#def mqtt_on_message(client, userdata, mqtt_msg):
#    """
#    This is the callback function at MQTT messages (un-matched topics) arrival.
#    ** On receiving this message, bypass to ROS.
#    """
#    print(mqtt_msg.topic + " " + str(mqtt_msg.payload))

def mqtt_advop_sync_cb(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving sync signal.
    ** On receiving this message, bypass to ROS.
    """
    rospy.loginfo('[MQTT] %s payload = %s', mqtt_msg.topic, mqtt_msg.payload)
    ROS_ADVOP_SYNC_PUB.publish(Empty())
    mqtt_publish_all_states()

def mqtt_advop_req_run_stop_cb(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving req_run_stop signal.
    ** On receiving this message, bypass to ROS.
    """
    rospy.loginfo("[MQTT] %s payload = %s", mqtt_msg.topic, mqtt_msg.payload)
    req_run_stop = mqtt_char_to_bool(mqtt_msg.payload)
    ROS_ADVOP_REQ_RUN_STOP_PUB.publish(req_run_stop) # "1" or "0"
    rospy.loginfo("[MQTT] Request to go" if req_run_stop else "[MQTT] Request to stop")
#------------------------------------------------------#
# end MQTT --> ROS




# ROS callbacks
#------------------------------------------------------------#
def ros_advop_run_state_cb(msg):
    """
    This is the callback function for run_state.
    ** On receiving this message, bypass to MQTT interface.
    """
    global VAR_ADVOP_RUN_STATE  # pylint: disable=global-statement
    VAR_ADVOP_RUN_STATE = msg.data # '1'
    rospy.loginfo("run_state = %s", str(VAR_ADVOP_RUN_STATE))
    if MQTT_CLIENT is None:
        return
    # Publish
    MQTT_CLIENT.publish(MQTT_ADVOP_RUN_STATE_PUBT,
                        payload=mqtt_bool_to_char(msg.data),
                        qos=2,
                        retain=False)

def ros_advop_sys_ready_cb(msg):
    """
    This is the callback function for sys_ready.
    ** On receiving this message, bypass to MQTT interface.
    """
    global VAR_ADVOP_SYS_READY  # pylint: disable=global-statement
    VAR_ADVOP_SYS_READY = msg.data # '1'
    rospy.loginfo("sys_ready = %s", str(VAR_ADVOP_SYS_READY))
    if MQTT_CLIENT is None:
        return
    set_timer_sys_ready()

def ros_advop_event_json_cb(msg):
    """
    This is the callback function for sys_ready.
    ** On receiving this message, bypass to MQTT interface.
    """
    event_json = msg.data #
    rospy.loginfo("[ADV_op_gateway] event_json = %s", event_json)
    if MQTT_CLIENT is None:
        return
    # Publish
    #MQTT_CLIENT.publish(mqtt_advop_event_json_pubT, payload=event_json, qos=2, retain=False)
    #
#------------------------------------------------------------#
# end ROS callbacks





def main():
    #
    global MQTT_CLIENT  # pylint: disable=global-statement
    # ROS
    rospy.init_node('ADV_op_gateway', anonymous=False)

    # Start timers
    set_timer_sys_ready()

    # ROS subscribers
    #-----------------------------#
    # Chose only one of them to receive the run state
    rospy.Subscriber("ADV_op/run_state", Bool, ros_advop_run_state_cb)
    #
    rospy.Subscriber("ADV_op/sys_ready", Bool, ros_advop_sys_ready_cb)
    rospy.Subscriber("ADV_op/event_json", String, ros_advop_event_json_cb)
    #-----------------------------#


    # MQTT
    #-------------------------------------------------------------------#
    MQTT_CLIENT = mqtt.Client()
    # Callback functions
    MQTT_CLIENT.on_connect = mqtt_on_connect
    # MQTT_CLIENT.on_message = mqtt_on_message # We don't need to process the message we don't want

    # Subscriber callbacks
    MQTT_CLIENT.message_callback_add(MQTT_ADVOP_SYNC_SUBT, mqtt_advop_sync_cb)
    MQTT_CLIENT.message_callback_add(MQTT_ADVOP_REQ_RUN_STOP_SUBT, mqtt_advop_req_run_stop_cb)

    # Connect
    is_connected = False
    while (not is_connected) and (not rospy.is_shutdown()):
        try:
            MQTT_CLIENT.connect(MQTT_BROKER_IP, 1883, 60)
            is_connected = True
            rospy.loginfo("[MQTT] Connected to broker.")
        except:  # pylint: disable=bare-except
            rospy.logwarn("[MQTT] Failed to connect to broker, keep trying.")
            time.sleep(1.0)
    # Start working
    MQTT_CLIENT.loop_start() # This start the actual work on another thread.

    # Publish the sync signal
    # Republish states
    mqtt_publish_all_states()
    #-------------------------------------------------------------------#

    rate = rospy.Rate(1.0) # Hz
    while not rospy.is_shutdown():
        mqtt_publish_all_states()
        try:
            rate.sleep()
        except:  # pylint: disable=bare-except
            # For ros time moved backward
            pass

    rospy.logwarn("[ADV_op_gateway] The ADV_op_gateway is going to close.")
    _timeout_handle_sys_ready(False)
    time.sleep(1.0)

if __name__ == '__main__':
    main()
