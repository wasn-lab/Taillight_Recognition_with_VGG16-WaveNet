import configparser
import json
import pprint
import time
import rospy
from heartbeat import Heartbeat
from itri_mqtt_client import ItriMqttClient
from ctrl_info03 import CtrlInfo03
from can_checker import CanChecker
from action_emitter import ActionEmitter
from status_level import OK, WARN, ERROR, FATAL


def _overall_status(module_states):
    return max(_["status"] for _ in module_states)


def _overall_status_str(module_states):
    mnames = [_["module"] for _ in module_states if _["status"] != OK]
    return "Misbehaving modules: {}".format(" ".join(mnames))


class FailSafeChecker(object):
    def __init__(self, cfg_ini, mqtt_ini):
        self.debug_mode = False
        rospy.init_node("FailSafeChecker")
        rospy.logwarn("Init FailSafeChecker")
        cfg = configparser.ConfigParser()
        self.modules = {}
        cfg.read(cfg_ini)
        self.latched_modules = []
        for module in cfg.sections():
            self.modules[module] = Heartbeat(
                module, cfg[module]["topic"],
                cfg[module].get("message_type", "Empty"),
                cfg[module].getfloat("fps_low"),
                cfg[module].getfloat("fps_high"),
                cfg[module].getboolean("inspect_message_contents"),
                cfg[module].getboolean("latch"))
            if cfg[module].getboolean("latch"):
                self.latched_modules.append(module)
        self.ctrl_info_03 = CtrlInfo03()
        self.can_checker = CanChecker()

        mqtt_cfg = configparser.ConfigParser()
        mqtt_cfg.read(mqtt_ini)
        self.mqtt_client = ItriMqttClient(mqtt_cfg["mqtt_broker"].get("fqdn", "127.0.0.1"))
        self.mqtt_topic = mqtt_cfg["mqtt_topics"]["fail_safe"]
        self.action_emitter = ActionEmitter()

        # counters for warn, error states. When the counter reaches 10,
        # change the state into next level (warn->error, error->fatal)
        self.warn_count = 0
        self.error_count = 0
        self.seq = 1

    def set_debug_mode(self, mode):
        self.debug_mode = mode

    def get_current_status(self):
        ret = {"states": self.ctrl_info_03.get_status_in_list(),
               "events": self.ctrl_info_03.get_events_in_list(),
               "seq": self.seq,
               "timestamp": time.time()}
        self.seq += 1
        ret["states"] += self.can_checker.get_status_in_list()
        ret["states"] += [self.modules[_].to_dict() for _ in self.modules]
        ret["status"] = _overall_status(ret["states"])
        ret["status_str"] = _overall_status_str(ret["states"])

        if (self.modules["3d_object_detection"].get_fps() +
                self.modules["LidarDetection"].get_fps()) == 0:
            ret["status"] = FATAL
            ret["status_str"] += "; Cam/Lidar detection offline at the same time"

        if ret["status"] == WARN:
            self.warn_count += 1
        else:
            self.warn_count = 0
        if self.warn_count > 10:
            ret["status"] = ERROR
            ret["status_str"] = "WARN states more than 10 seconds"

        if ret["status"] == ERROR:
            self.error_count += 1
        else:
            self.error_count = 0
        if self.error_count > 10:
            ret["status"] = FATAL
            ret["status_str"] = "ERROR states more than 10 seconds"

        return ret

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            for module in self.latched_modules:
                self.modules[module].update_latched_message()
            current_status = self.get_current_status()
            if self.debug_mode:
                pprint.pprint(current_status)
            if current_status["status"] != OK:
                self.action_emitter.backup_rosbag(current_status["status_str"])
            self.mqtt_client.publish(self.mqtt_topic, json.dumps(current_status))
            rate.sleep()
