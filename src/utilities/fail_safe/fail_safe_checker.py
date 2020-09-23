import configparser
import rospy
import json
from collections import Counter
from heartbeat import Heartbeat
from itri_mqtt_client import ItriMqttClient


def _overall_status(module_states):
    counter = Counter([_["status"] for _ in module_states])
    for key in ["FATAL", "ERROR", "WARN"]:
        if counter[key] > 0:
            return key
    return "OK"


def _overall_status_str(module_states):
    mnames = [_["module"] for _ in module_states if _["status"] != "OK"]
    return "Misbehaving moudles: {}".format(" ".join(mnames))


class FailSafeChecker():
    def __init__(self, cfg_ini, mqtt_ini):
        rospy.init_node("FailSafeChecker")
        rospy.logwarn("Init FailSafeChecker")
        cfg = configparser.ConfigParser()
        self.heartbeats = {}
        cfg.read(cfg_ini)
        self.latched_modules = []
        for module in cfg.sections():
            self.heartbeats[module] = Heartbeat(
                module, cfg[module]["topic"],
                cfg[module].get("message_type", "Empty"),
                cfg[module].getfloat("fps_low"),
                cfg[module].getfloat("fps_high"),
                cfg[module].getboolean("inspect_message_contents"),
                cfg[module].getboolean("latch"))
            if cfg[module].getboolean("latch"):
                self.latched_modules.append(module)

        mqtt_cfg = configparser.ConfigParser()
        mqtt_cfg.read(mqtt_ini)
        self.mqtt_client = ItriMqttClient(mqtt_cfg["mqtt_broker"].get("fqdn", "127.0.0.1"))
        self.mqtt_topic = mqtt_cfg["mqtt_topics"]["fail_safe"]

    def get_current_status(self):
        ret = {"states": [self.heartbeats[_].to_dict() for _ in self.heartbeats],
               "events": [],
               }
        ret["status"] = _overall_status(ret["states"])
        ret["status_str"] = _overall_status_str(ret["states"])
        return ret

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            for module in self.latched_modules:
                self.heartbeats[module].update_latched_message()
            jdata = json.dumps(self.get_current_status())
            self.mqtt_client.publish(self.mqtt_topic, jdata)
            rate.sleep()
