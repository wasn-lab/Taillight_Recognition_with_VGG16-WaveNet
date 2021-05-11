# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
from msgs.msg import Flag_Info
from status_level import OK, WARN, FATAL
from ctrl_info03 import CtrlInfo03, BrakeStatus


class CtrlInfo03Test(unittest.TestCase):
    def test_1(self):
        self.node = rospy.init_node("ctrl_info_test_node", anonymous=True)
        self.ctrl03 = CtrlInfo03()

        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 0)
        msg = Flag_Info()

        msg.Dspace_Flag05 = float(BrakeStatus.Y_MANUAL_BRAKE)
        self.ctrl03._cb(msg)

        events = self.ctrl03.get_events_in_list()
        print(events)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], FATAL)


    @unittest.skip("Manually enabled test item")
    def test_run(self):
        self.sender.run()


if __name__ == "__main__":
    unittest.main()
