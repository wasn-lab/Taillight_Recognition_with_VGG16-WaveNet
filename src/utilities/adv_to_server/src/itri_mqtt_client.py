# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import paho.mqtt.client as mqtt

class ItriMqttClient():
    def __init__(self, fqdn, port):
        self.fqdn = fqdn
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        print("Try connecting to MQTT server {}, port {}".format(fqdn, port))
        self.client.connect(fqdn, int(port), 60)
        print("Successfuly connect to MQTT server")
        self.client.loop_start()

    def __del__(self):
        self.client.loop_stop()

    def on_message(self, _client, _userdata, message):
        print("message received ", str(message.payload.decode("utf-8")))
        print("message topic=", message.topic)
        print("message qos=", message.qos)
        print("message retain flag=", message.retain)

    def on_connect(self, _client, _userdata, _flags, _rc):
        print("Connect to mqtt broker {}".format(self.fqdn))

    def publish(self, topic, payload):
        return self.client.publish(topic, payload=payload, qos=2, retain=False)

    def subscribe(self, topic, callback, qos=2):
        """
        qos -- At most once (0), At least once (1), Exactly once (2).
        """
        self.client.subscribe(topic, qos)
        self.client.on_message = callback
