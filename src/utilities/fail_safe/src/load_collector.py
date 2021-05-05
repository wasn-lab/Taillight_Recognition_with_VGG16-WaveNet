# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Aggregate cpu/gpu loads and publish it
"""
import datetime
import json
import rospy
from std_msgs.msg import String
from car_model_helper import get_car_model_as_str
from itri_mqtt_client import ItriMqttClient
from status_level import OK, WARN

_MQTT_VEHICLE_SYSTEM_LOADS_TOPIC = "vehicle/report/system_loads"
SYSTEM_LOADS_TOPIC = "/vehicle/report/system_loads"

class LoadCollector(object):
    def __init__(self, mqtt_fqdn, mqtt_port):
        rospy.init_node("LoadCollector")
        car_model = get_car_model_as_str()
        self.ipcs = []
        if car_model in ["B1", "B1_V2", "B1_V3", "C1"]:
            self.ipcs = ["lidar", "camera", "localization", "xavier"]
        else:
            rospy.logerr("Unrecognized car_model: %s", car_model)

        rospy.logwarn("Init LoadCollector node")
        self.records, self.timestamps = self.setup_records()

        for ipc in self.ipcs:
            ipc = ipc.replace("-", "_")
            topic = "/vehicle/report/{}/load".format(ipc)
            rospy.Subscriber(topic, String, callback=self._cb, queue_size=1)
            rospy.logwarn("Subscribe %s", topic)

        self.load_publisher = rospy.Publisher(SYSTEM_LOADS_TOPIC, String, queue_size=1)
        rospy.logwarn("Publish on %s", SYSTEM_LOADS_TOPIC)
        self.mqtt_client = ItriMqttClient(mqtt_fqdn, mqtt_port)

    def setup_records(self):
        now = datetime.datetime.now()
        self.records = {}
        self.timestamps = {ipc: now for ipc in self.ipcs}
        for ipc in self.ipcs:
            self.records[ipc] = {"cpu_load": 0,
                                 "gpu_load": 0,
                                 "status": OK,
                                 "status_str": ""}
        return self.records, self.timestamps

    def _cb(self, msg):
        jdata = json.loads(msg.data)
        ipc = jdata["hostname"]

        self.timestamps[ipc] = datetime.datetime.now()
        self.records[ipc]["cpu_load"] = jdata["cpu_load"]
        self.records[ipc]["gpu_load"] = jdata["gpu_load"]
        self.records[ipc]["status"] = OK
        status_str = ""

        if jdata["cpu_load"] >= jdata["cpu_load_threshold"]:
            self.records[ipc]["status"] = WARN
            status_str = "high cpu load: " + str(self.records[ipc]["cpu_load"])[:5]

        if jdata["gpu_load"] >= 99:
            temp = "high gpu load: " + str(self.records[ipc]["gpu_load"])[:5]
            if status_str:
                status_str += ", " + temp
            else:
                status_str = temp
        self.records[ipc]["status_str"] = status_str

    def get_current_loads(self, now=None):
        """
        Return the latest loads.
        If we do not get cpu/gpu loads in recent 3 seconds, then show it as NA
        """
        if now is None:
            now = datetime.datetime.now()
        for ipc in self.ipcs:
            delta = now - self.timestamps[ipc]
            if delta.total_seconds() > 3:
                rospy.logwarn("%s: not receiving cpu/gpu loads", ipc)
                self.records[ipc]["status"] = WARN
                self.records[ipc]["status_str"] = "not receiving cpu/gpu loads"
        return self.records

    def run(self):
        rate = rospy.Rate(1)  # FPS: 1
        while not rospy.is_shutdown():
            loads = self.get_current_loads()
            jloads = json.dumps(loads);
            self.load_publisher.publish(jloads)
            self.mqtt_client.publish(_MQTT_VEHICLE_SYSTEM_LOADS_TOPIC, jloads)
            rate.sleep()
