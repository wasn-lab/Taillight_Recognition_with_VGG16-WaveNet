import configparser
import rospy
from heartbeat import Heartbeat

class FailSafeChecker():
    def __init__(self, cfg_ini):
        rospy.init_node("FailSafeChecker")
        rospy.logwarn("Init FailSafeChecker")
        cfg = configparser.ConfigParser()
        cfg.read(cfg_ini)
        self.heartbeats = {}
        for module in cfg.sections():
            self.heartbeats[module] = Heartbeat(
                module, cfg[module]["topic"],
                cfg[module].get("message_type", "Empty"),
                cfg[module].getfloat("fps_low"),
                cfg[module].getfloat("fps_high"))

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
