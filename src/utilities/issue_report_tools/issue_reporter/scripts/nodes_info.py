import configparser

import rospy
from std_msgs.msg import Empty
from heartbeat import Heartbeat

def main():
    rospy.init_node("FailSafeChecker")
    cfg = configparser.ConfigParser()
    cfg.read("nodes_info.ini")
    heartbeats = {}
    for module in cfg.sections():
        heartbeats[module] = Heartbeat(module, cfg[module]["topic"],
                                       cfg[module].get("message_type", "Empty"),
                                       cfg[module].getfloat("fps_low"),
                                       cfg[module].getfloat("fps_high"))

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == "__main__":
    main()
