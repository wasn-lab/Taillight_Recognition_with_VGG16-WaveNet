import configparser
import pprint
import rospy
from heartbeat import Heartbeat

class FailSafeChecker():
    def __init__(self, cfg_ini):
        rospy.init_node("FailSafeChecker")
        rospy.logwarn("Init FailSafeChecker")
        cfg = configparser.ConfigParser()
        self.heartbeats = {}
        cfg.read(cfg_ini)
        for module in cfg.sections():
            self.heartbeats[module] = Heartbeat(
                module, cfg[module]["topic"],
                cfg[module].get("message_type", "Empty"),
                cfg[module].getfloat("fps_low"),
                cfg[module].getfloat("fps_high"),
                cfg[module].getboolean("inspect_message_contents"))

    def get_node_status(self):
        return [self.heartbeats[_].to_dict() for _ in self.heartbeats]

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            pprint.pprint(self.get_node_status())
            rate.sleep()
