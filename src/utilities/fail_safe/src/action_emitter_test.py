import unittest
import time
import rospy
from action_emitter import ActionEmitter


class ActionEmitterTest(unittest.TestCase):
    def setUp(self):
        self.obj = ActionEmitter()

    def test_1(self):
        time.sleep(1)  # wait for connecting to listener
        self.obj.backup_rosbag(reason="unittest at {}".format(time.time()))


if __name__ == "__main__":
    rospy.init_node("ActionEmitterTest")
    rospy.logwarn("Init ActionEmitterTest")
    unittest.main()
