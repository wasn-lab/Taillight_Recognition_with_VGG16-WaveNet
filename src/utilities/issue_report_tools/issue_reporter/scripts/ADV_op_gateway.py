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


# ROS Publishers
msg_recorder_trigger_pub = rospy.Publisher('/REC/req_backup', Empty, queue_size=10)



# MQTT
#------------------------------------------------------------#
mqtt_client = None
mqtt_broker = "127.0.0.1" # localhost
# MQTT topics
mqtt_msg_topic_namespace = 'ADV_op'

# Subscribed topics (MQTT --> ROS)
#------------------------#
# Alive signal
mqtt_REC_req_backup_subT = mqtt_msg_topic_namespace + '/req_backup'

# The subscription list
# element: (topic, QoS)
mqtt_subscription_list = list()
# NOTE: since that we are subscrbing to local broker, the loading is tiny for QoS=2
mqtt_subscription_list.append( (mqtt_REC_req_backup_subT, 2) )
#------------------------#

# Published topics (MQTT --> dock)
#------------------------#
mqtt_REC_report_pubT = mqtt_msg_topic_namespace + '/trigger_report' #
#------------------------#

#------------------------------------------------------------#
# end MQTT





# MQTT --> ROS
#------------------------------------------------------#
# The callback for when the client receives a CONNACK response from the server.
def mqtt_on_connect(client, userdata, flags, rc):
    """
    This is the callback function at connecting the MQTT client.
    """
    global mqtt_luggage_dock_namespace

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

def mqtt_REC_req_backup_CB(client, userdata, mqtt_msg):
    """
    This is the callback function at receiving req_backup signal.
    ** On receiving this message, bypass to ROS.
    """
    print( 'payload = "%s"' % mqtt_msg.payload )
    msg_recorder_trigger_pub.publish( Empty() )
#------------------------------------------------------#
# end MQTT --> ROS

# ROS callbacks
#------------------------------------------------------------#
def GUI2_state_CB(msg):
    global GUI_state, GUI_state_seq
    GUI_state = msg
    GUI_state_seq += 1
    print("[GUI-gateway] GUI state recieved.")

def REC_report_CB(msg):
    """
    This is the callback function for reporting trigger event.
    ** On receiving this message, bypass to MQTT interface.
    """
    if mqtt_client is None:
        return
    # Publish
    payload_ = msg.data # '1'
    mqtt_client.publish(mqtt_REC_report_pubT, payload=payload_, qos=2, retain=False)
#------------------------------------------------------------#
# end ROS callbacks




# Wait for ROS terminating signal to close the HTTP server
def wait_for_close():
    pass


def main():
    #
    global mqtt_client
    # ROS
    rospy.init_node('GUI_gateway', anonymous=True)
    rospy.Subscriber("GUI2/state", GUI2_op, GUI2_state_CB)
    rospy.Subscriber("REC/trigger_report", String, REC_report_CB)

    _t = threading.Thread(target=wait_for_close)
    # _t.daemon = True
    _t.start()


    # MQTT
    #-------------------------------------------------------------------#
    mqtt_client = mqtt.Client()
    # Callback functions
    mqtt_client.on_connect = mqtt_on_connect
    # mqtt_client.on_message = mqtt_on_message # We don't need to process the message we don't want

    # Subscriber callbacks
    mqtt_client.message_callback_add(mqtt_REC_req_backup_subT, mqtt_REC_req_backup_CB)

    # Connect
    mqtt_client.connect(mqtt_broker, 1883, 60)
    # Start working
    mqtt_client.loop_start() # This start the actual work on another thread.

    # Publish the sync signal
    # mqtt_client.publish(mqtt_sync_pubT, payload='1', qos=2, retain=False)
    #-------------------------------------------------------------------#

    while not rospy.is_shutdown():
        rospy.sleep(0.5)

    print("End of GUI_gateway")

if __name__ == '__main__':
    main()
