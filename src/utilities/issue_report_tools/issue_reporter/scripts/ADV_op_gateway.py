#!/usr/bin/env python
import random
import string
import json
import threading
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
#---------------------------#



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





# MQTT --> ROS
#------------------------------------------------------#
def mqtt_bool_to_char(bool_in):
    """
    """
    return ("1" if bool_in else "0")

def mqtt_char_to_bool(char_in):
    """
    """
    return (char_in == "1")

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
    print( 'payload = "%s"' % mqtt_msg.payload )
    ros_advop_sync_pub.publish( Empty() )
    mqtt_publish_all_states()

def mqtt_advop_req_run_stop_CB(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving req_run_stop signal.
    ** On receiving this message, bypass to ROS.
    """
    print( 'payload = "%s"' % mqtt_msg.payload )
    ros_advop_req_run_stop_pub.publish( mqtt_char_to_bool(mqtt_msg.payload) ) # "1" or "0"
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
    print("var_advop_run_state = %s" % str(var_advop_run_state))
    if mqtt_client is None:
        return
    # Publish
    mqtt_client.publish(mqtt_advop_run_state_pubT, payload=mqtt_bool_to_char(msg.data), qos=2, retain=False)

def ros_advop_sys_ready_CB(msg):
    """
    This is the callback function for sys_ready.
    ** On receiving this message, bypass to MQTT interface.
    """
    global var_advop_sys_ready
    var_advop_sys_ready = msg.data # '1'
    print("var_advop_sys_ready = %s" % str(var_advop_sys_ready))
    if mqtt_client is None:
        return
    # Publish
    mqtt_client.publish(mqtt_advop_sys_ready_pubT, payload=mqtt_bool_to_char(msg.data), qos=2, retain=False)
#------------------------------------------------------------#
# end ROS callbacks





def main():
    #
    global mqtt_client
    # ROS
    rospy.init_node('ADV_op_gateway', anonymous=True)
    rospy.Subscriber("ADV_op/run_state", Bool, ros_advop_run_state_CB)
    rospy.Subscriber("ADV_op/sys_ready", Bool, ros_advop_sys_ready_CB)


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
    mqtt_client.connect(mqtt_broker, 1883, 60)
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
        rate.sleep()

    print("End of ADV_op_gateway")

if __name__ == '__main__':
    main()
