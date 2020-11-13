import unittest
import rospy
import pprint
from pedcross_alert import PedCrossAlert


class PedCrossAlertTest(unittest.TestCase):
    def setUp(self):
        self.obj = PedCrossAlert()

    def test_1(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            pprint.pprint(self.obj.get_events_in_list())
            pprint.pprint(self.obj.get_status_in_list())
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("PedCrossAlertTestNode")
    unittest.main()
