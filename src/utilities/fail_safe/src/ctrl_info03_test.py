# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
from msgs.msg import Flag_Info
from status_level import OK, WARN, FATAL
from ctrl_info03 import CtrlInfo03, BrakeStatus


class CtrlInfo03Test(unittest.TestCase):
    def setUp(self):
        self.node = rospy.init_node("ctrl_info_test_node", anonymous=True)
        self.ctrl03 = CtrlInfo03()
        self.msg = Flag_Info()

    def test_disengage_events(self):
        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 0)

        # receive manual break
        self.msg.Dspace_Flag05 = float(BrakeStatus.Y_MANUAL_BRAKE)
        self.ctrl03._cb(self.msg)

        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], FATAL)

        # do not report the same disengage event 
        for _ in range(100):
            self.ctrl03._cb(self.msg)
        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 0)

        self.ctrl03._cb(self.msg)
        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 0)

    def test_disengage_events_2(self):
        # receive manual break
        self.msg.Dspace_Flag05 = float(BrakeStatus.Y_MANUAL_BRAKE)
        self.ctrl03._cb(self.msg)

        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], FATAL)

        # No disengage event in this round
        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 0)

        # View the disengage event as a new one and report it.
        self.ctrl03._cb(self.msg)
        events = self.ctrl03.get_events_in_list()
        self.assertEqual(len(events), 1)

if __name__ == "__main__":
    unittest.main()
