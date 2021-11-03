# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
import pprint
from status_level import OK, WARN, ERROR, FATAL, STATUS_CODE_TO_STR
from fail_safe_checker import aggregate_event_status


class FailSafeCheckerTest(unittest.TestCase):
    def test_aggregate_event_status_1(self):
        status = OK
        status_str = ""
        events = [{"module": "pedcross_event",
                   "status": WARN,
                   "status_str": "pedestrian is crossing"}]
        status, status_str = aggregate_event_status(status, status_str, events)
        self.assertEqual(status, WARN)
        self.assertEqual(status_str, "pedestrian is crossing")

    def test_aggregate_event_status_2(self):
        status = ERROR
        status_str = "Misheaving modules: CAN"
        events = [{"module": "pedcross_event",
                   "status": WARN,
                   "status_str": "pedestrian is crossing"}]
        status, status_str = aggregate_event_status(status, status_str, events)
        self.assertEqual(status, ERROR)
        self.assertEqual(status_str, "Misheaving modules: CAN; pedestrian is crossing")

    def test_aggregate_event_status_3(self):
        status = ERROR
        status_str = "Misheaving modules: CAN"
        events = [{"module": "pedcross_event",
                   "status": WARN,
                   "status_str": "pedestrian is crossing"},
                  {"module": "disengage_event",
                   "status": FATAL,
                   "status_str": "Disengage: Driver manually press brake pedals!"}
                 ]
        status, status_str = aggregate_event_status(status, status_str, events)
        self.assertEqual(status, FATAL)
        self.assertEqual(status_str, 
                         ("Misheaving modules: CAN; pedestrian is crossing; "
                          "Disengage: Driver manually press brake pedals!"))


if __name__ == "__main__":
    unittest.main()
