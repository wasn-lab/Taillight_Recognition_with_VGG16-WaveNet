#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from itri_mqtt_client import ItriMqttClient
import json
from datetime import datetime
from threading import Timer


message_buffer_size = 30
message_buffer = list()
mqtt_client = ItriMqttClient("60.250.196.127", 1884)
timout = 5 # second


def mqtt_publish():
    global mqtt_client
    if len(message_buffer) > 0:
        mileage_json_str = gen_json()
        del message_buffer[:]
        mqtt_client.publish("mileage", mileage_json_str)


timer = Timer(timout, mqtt_publish)


def gen_json():
    global message_buffer
    plate = rospy.get_param("/south_bridge/license_plate_number", "ITRI-ADV")
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rospy.loginfo(message_buffer)
    json_obj = {"type": "M8.2.VK006.1", "deviceid": plate, "receivetime": dt_string, "mileage_info": message_buffer}
    return json.dumps(json_obj)


def callback(data):
    global message_buffer
    global message_buffer_size
    global timer

    timer.cancel()
    timer = Timer(timout, mqtt_publish)
    timer.start()

    mileage_json_obj = json.loads(data.data)
    rospy.loginfo(json.dumps(mileage_json_obj))
    message_buffer.append(mileage_json_obj)
    rospy.loginfo("============ message buffer len: %s/%s ============",len(message_buffer), message_buffer_size)

    if len(message_buffer) == message_buffer_size:
        mqtt_publish()
        
    
def listener():
    rospy.init_node('mileage_listener', anonymous=True)
    rospy.Subscriber("/mileage/relative_mileage", String, callback)
    rospy.spin()


if __name__ == '__main__':
    print("test")
    listener()
