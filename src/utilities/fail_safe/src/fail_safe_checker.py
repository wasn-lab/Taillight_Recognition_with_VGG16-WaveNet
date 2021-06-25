# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import configparser
import json
import pprint
import rospy
from std_msgs.msg import String, Bool
from heartbeat import Heartbeat
from itri_mqtt_client import ItriMqttClient
from ctrl_info02 import CtrlInfo02
from ctrl_info03 import CtrlInfo03
from can_checker import CanChecker
from pedcross_alert import PedCrossAlert
from system_load_checker import SystemLoadChecker
from action_emitter import ActionEmitter
from status_level import OK, WARN, ERROR, FATAL, STATUS_CODE_TO_STR
from sb_param_utils import get_vid
from issue_reporter import IssueReporter, generate_issue_description
from timestamp_utils import get_timestamp_mot

_MQTT_FAIL_SAFE_TOPIC = "/fail_safe"  # To be removed in the future
_MQTT_FAIL_SAFE_STATUS_TOPIC = "vehicle/report/itri/fail_safe_status"
_MQTT_SYS_READY_TOPIC = "ADV_op/sys_ready"


def _overall_status(module_states):
    return max(_["status"] for _ in module_states)


def _overall_status_str(module_states):
    mnames = [_["module"] for _ in module_states if _["status"] != OK]
    return "Misbehaving modules: {}".format(" ".join(mnames))

def aggregate_event_status(status, status_str, events):
    """
    Args:
    status(int) -- overall status of states
    status_str(str) -- aggregated strs of states

    Return:
    status(int) -- Highest level between |status| and |events|
    status_str(str) -- Aggregated status_str with |events|
    """
    for event in events:
        status = max(status, event["status"])
        if event["status"] != OK:
            if status_str:
                status_str += "; " + event["status_str"]
            else:
                status_str = event["status_str"]
    return status, status_str


class FailSafeChecker(object):
    def __init__(self, heartbeat_ini, mqtt_fqdn, mqtt_port):
        self.debug_mode = False
        self.vid = get_vid()  # vehicle id
        rospy.init_node("FailSafeChecker")
        rospy.logwarn("Init FailSafeChecker")
        cfg = configparser.ConfigParser()
        self.modules = {}
        cfg.read(heartbeat_ini)
        self.latched_modules = []
        for module in cfg.sections():
            self.modules[module] = Heartbeat(
                module, cfg[module]["topic"],
                cfg[module].get("message_type", "Empty"),
                cfg[module].getfloat("fps_low"),
                cfg[module].getfloat("fps_high"),
                cfg[module].getboolean("inspect_message_contents"),
                cfg[module].getboolean("latch"),
                cfg[module].get("sensor_type", None),
                cfg[module].get("sensor_uid", None))
            enable = cfg[module].getboolean("enable", True)
            self.modules[module].set_enabled(enable)
            if cfg[module].getboolean("latch"):
                self.latched_modules.append(module)
        self.ctrl_info_03 = CtrlInfo03()
        self.ctrl_info_02 = CtrlInfo02()
        self.pedcross_alert = PedCrossAlert()
        self.system_load_checker = SystemLoadChecker()
        self.can_checker = CanChecker()
        self.issue_reporter = IssueReporter()

        self.mqtt_client = ItriMqttClient(mqtt_fqdn, mqtt_port)
        self.action_emitter = ActionEmitter()
        self.sensor_status_publisher = rospy.Publisher(
            "/vehicle/report/itri/sensor_status", String, queue_size=1000)
        self.fail_safe_status_publisher = rospy.Publisher(
            "/vehicle/report/itri/fail_safe_status", String, queue_size=1000)
        self.self_driving_mode_publisher = rospy.Publisher(
            "/vehicle/report/itri/self_driving_mode", Bool, queue_size=2)
        self.sys_ready_publisher = rospy.Publisher(
            "/ADV_op/sys_ready", Bool, queue_size=1000)

        # counters for warn, error states. When the counter reaches 10,
        # change the state into next level (warn->error, error->fatal)
        self.warn_count = 0
        self.error_count = 0
        self.seq = 1

    def _get_ego_speed(self):
        return self.modules["veh_info"].get_ego_speed()

    def set_debug_mode(self, mode):
        self.debug_mode = mode

    def get_current_status(self):
        """Collect states from the components of the car"""
        ego_speed = self._get_ego_speed()

        ret = {"states": self.ctrl_info_03.get_status_in_list(),
               "events": self.ctrl_info_03.get_events_in_list(),
               "seq": self.seq,
               "timestamp": get_timestamp_mot()}
        self.seq += 1
        ret["states"] += self.can_checker.get_status_in_list()
        ret["states"] += [self.modules[_].to_dict() for _ in self.modules]
        # pedcross is still under heavy development
        ret["states"] += self.pedcross_alert.get_status_in_list()
        ret["states"] += self.system_load_checker.get_status_in_list()
        status = _overall_status(ret["states"])
        status_str = _overall_status_str(ret["states"])

        if ego_speed > 0:
            ret["events"] += self.pedcross_alert.get_events_in_list()

        if (self.modules["3d_object_detection"].get_fps() +
                self.modules["LidarDetection"].get_fps()) == 0:
            status = FATAL
            status_str += "; Cam/Lidar detection offline at the same time"

        if status == OK:
            self.warn_count = 0
            self.error_count = 0

        if status == WARN:
            self.warn_count += 1
        else:
            self.warn_count = 0
        if self.warn_count > 10 and ego_speed > 0:
            status = ERROR
            status_str += "; WARN states more than 10 seconds"

        if status == ERROR:
            self.error_count += 1
        else:
            self.error_count = 0
        if self.error_count > 10:
            status = FATAL
            status_str += "; ERROR states more than 10 seconds"

        status, status_str = aggregate_event_status(status, status_str, ret["events"])
        ret["status"] = status
        ret["status_str"] = status_str
        self._publish_sys_ready(status)

        return ret

    def _publish_sys_ready(self, status):
        if status == FATAL:
            # force stop self-driving mode
            self.sys_ready_publisher.publish(False)
            self.mqtt_client.publish(_MQTT_SYS_READY_TOPIC, "0")
        else:
            self.sys_ready_publisher.publish(True)
            self.mqtt_client.publish(_MQTT_SYS_READY_TOPIC, "1")

    def _get_all_sensor_status(self):
        docs = {"vid": self.vid,
                "camera": [],
                "gps": [],
                "lidar": [],
                "radar": []}
        for mod_name in self.modules:
            module = self.modules[mod_name]
            if module.sensor_type is None:
                continue
            doc = module.get_sensor_status()
            docs[module.sensor_type].append(doc)
        return docs

    def is_self_driving(self):
        return self.ctrl_info_02.is_self_driving()

    def post_issue_if_necessary(self, current_status):
        if not self.is_self_driving():
            if current_status["status"] != OK:
                rospy.logwarn("Do not post issue in non-self-driving mode")
            return

        if not rospy.get_param("/fail_safe/should_post_issue", True):
            if current_status["status"] != OK:
                rospy.logwarn("Do not post issue due to /fail_safe/should_post_issue is False")
            return

        for doc in current_status["events"]:
            if doc["status"] != OK:
                summary = "[Auto Report] {}: {}".format(
                    doc["module"], doc["status_str"])
                desc = generate_issue_description(
                    doc["status"], doc["status_str"], current_status["timestamp"])
                self.issue_reporter.post_issue(summary, desc)
                return
        if current_status["status"] != OK:
            summary = "[Auto Report] {}".format(
                current_status["status_str"])
            desc = generate_issue_description(
                current_status["status"], current_status["status_str"], current_status["timestamp"])
            self.issue_reporter.post_issue(summary, desc)

    def run(self):
        """Send out aggregated info to backend server every second."""
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            for module in self.latched_modules:
                self.modules[module].update_latched_message()
            current_status = self.get_current_status()
            sensor_status = self._get_all_sensor_status()

            rospy.logwarn("status: %s -- %s",
                          STATUS_CODE_TO_STR[current_status["status"]],
                          current_status["status_str"])
            if self.debug_mode:
                # pprint.pprint(sensor_status)
                pprint.pprint(current_status)
            if current_status["status"] != OK and self.is_self_driving():
                self.action_emitter.backup_rosbag(current_status["status_str"])

            self.post_issue_if_necessary(current_status)
            current_status_json = json.dumps(current_status)
            self.mqtt_client.publish(_MQTT_FAIL_SAFE_TOPIC, current_status_json)
            self.mqtt_client.publish(_MQTT_FAIL_SAFE_STATUS_TOPIC, current_status_json)
            self.fail_safe_status_publisher.publish(current_status_json)
            self.sensor_status_publisher.publish(json.dumps(sensor_status))
            self.self_driving_mode_publisher.publish(Bool(self.is_self_driving()))

            if self.warn_count + self.error_count > 0:
                rospy.logwarn("warn_count: %d, error_count: %d",
                              self.warn_count, self.error_count)
            rate.sleep()
