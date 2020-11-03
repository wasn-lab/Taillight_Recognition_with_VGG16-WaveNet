import configparser
import argparse
import os
import rospkg
import rospy
from std_msgs.msg import Empty

class MockHeartbeatGenerator():
    def __init__(self, hearbeat_ini):
        rospy.init_node("MockHeartbeatGenerator")
        cfg = configparser.ConfigParser()
        cfg.read(hearbeat_ini)
        self.publishers = []
        for module in cfg.sections():
            topic = cfg[module]["topic"]
            if "heartbeat" in topic:
                self.publishers.append(rospy.Publisher(topic, Empty, queue_size=1))

    def run(self):
        rate = rospy.Rate(10)
        cnt = 0
        while not rospy.is_shutdown():
            if cnt == 0:
                rospy.logwarn("publish heartbeat")
            cnt += 1
            if cnt == 100:
                cnt = 0
            for pub in self.publishers:
                pub.publish()
            rate.sleep()

def main():
    pkg_dir = rospkg.RosPack().get_path("fail_safe")
    src_dir = os.path.join(pkg_dir, "src")
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat-ini", default=os.path.join(src_dir, "heartbeat.ini"))
    args = parser.parse_args()
    gnr = MockHeartbeatGenerator(args.heartbeat_ini)
    gnr.run()


if __name__ == "__main__":
    main()
