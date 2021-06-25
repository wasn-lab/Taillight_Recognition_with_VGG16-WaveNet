# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
from paho.mqtt.client import MQTTMessage
from pad_issue_reporter import _parse_payload

class PadIssueTest(unittest.TestCase):
    def test__parse_payload(self):
        msg = MQTTMessage()
        msg.payload = '{"modules":["veh_info","recorder","CAN","XByWire","Tracking3D","pedcross","other"]}'
        msg.topic = "fail_safe/req_report_issue"
        msg.qos = 0
        jdata = _parse_payload(msg.payload)
        self.assertTrue("other" in jdata["modules"])


if __name__ == "__main__":
    unittest.main()
