import configparser
import logging
import paho.mqtt.client as mqtt

class ItriMqttClient():
    def __init__(self, fqdn):
        self.fqdn = fqdn
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish
        self.client.connect(fqdn, 1883, 60)
        self.client.loop_start()

    def __del__(self):
        self.client.loop_stop()

    def on_publish(self, client, userdata, mid):
        return

    def on_connect(self, client, userdata, flags, rc):
        print("Connect to mqtt broker {}".format(self.fqdn))
        client.subscribe("$SYS/#")

    def publish(self, topic, payload):
        self.client.publish(topic, payload=payload, qos=2, retain=False)
        return 0
