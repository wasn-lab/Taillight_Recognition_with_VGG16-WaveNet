# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
import pprint
from system_load_checker import SystemLoadChecker


class SystemLoadCheckerTest(unittest.TestCase):
    def setUp(self):
        self.obj = SystemLoadChecker()

    def test_1(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            pprint.pprint(self.obj.get_status_in_list())
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("SystemLoadCheckerNode")
    unittest.main()
