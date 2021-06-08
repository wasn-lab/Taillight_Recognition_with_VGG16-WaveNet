#!/usr/bin/env python
from sqlite3 import Error

import rospy
from std_msgs.msg import String
from itri_mqtt_client import ItriMqttClient
import json
import sqlite3
import datetime
import rospkg

from os.path import expanduser
home = expanduser("~")

mqtt_client = ItriMqttClient("60.250.196.127", 1884)
TOPIC = "events"


def create_table(connection):
    if connection is not None:
        sql_create_events_table = \
            """ CREATE TABLE IF NOT EXISTS events (
                    time TEXT NOT NULL,
                    status INTEGER,
                    status_str TEXT,
                    module TEXT
            ); """
        try:
            c = connection.cursor()
            c.execute(sql_create_events_table)
        except Error as e:
            # rospy.logerr("Error! cannot create table.")
            print("Error! cannot create table.")
    else:
        # rospy.logerr("Error! cannot create the database connection.")
        print("Error! cannot create the database connection.")

def get_path():
       rospack = rospkg.RosPack()
       rospack.list()
       pk_path = rospack.get_path('events_to_server')
       # print pk_path
       return pk_path


def write_to_db(json_str):
    connection = sqlite3.connect(home + "/adv.db")
    create_table(connection)
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_obj = json.loads(json_str) 
    events = json_obj["events"]
    for event in events:
        x = (time, str(event["status"]), event["status_str"], event["module"] )
        sql = '''insert into events values(?,?,?,?)'''
        connection.execute(sql, x)
        connection.commit()
    connection.close()


def mqtt_publish(fail_safe_json_obj):
    global mqtt_client
    if len(fail_safe_json_obj["events"]) > 0:
        event_json_str = gen_json(fail_safe_json_obj["events"])
        write_to_db(event_json_str)
        result = mqtt_client.publish(TOPIC, event_json_str)
        print("publish result: ",  "success" if result[0] == 0 else "fail")
    else:
        print("No events.")

    


def gen_json(events):
    plate = rospy.get_param("/south_bridge/license_plate_number", "ITRI-ADV")
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_obj = {"type": "M8.2.VK003.2", "deviceid": plate, "timestamp": dt_string,  "events": events}
    return json.dumps(json_obj)


def callback(data):
    # print(data.data)
    fail_safe_json_obj = json.loads(data.data)
    mqtt_publish(fail_safe_json_obj)


def listener():
    rospy.init_node('mileage_listener', anonymous=True)
    rospy.Subscriber("/vehicle/report/itri/fail_safe_status", String, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
