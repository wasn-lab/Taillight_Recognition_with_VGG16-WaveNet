#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import time
import json
import rospy
from itri_mqtt_client import ItriMqttClient
from status_level import FATAL
from issue_reporter import IssueReporter, generate_issue_description

_MQTT_REQ_REPORT_ISSUE_TOPIC = "fail_safe/req_report_issue"


def _parse_payload(payload):
    return json.loads(payload)


def _post_issue(_client, _userdata, message):
    # print("message received ", str(message.payload.decode("utf-8")))
    # print("message topic=", message.topic)
    # print("message qos=", message.qos)
    # print("message retain flag=", message.retain)
    issue_reporter = IssueReporter()
    jdata = _parse_payload(message.payload)
    summary = u"[Auto Report][From PAD] 異常模組：{}".format(u",".join(jdata.get("modules", ["unknown"])))

    timestamp = time.time()
    description = generate_issue_description(
        FATAL, u"請連至以下後台網址取得當時狀態", timestamp)
    rospy.logwarn("Use click the post issue button!")
    print(summary)
    print(description)
    issue_reporter.post_issue(summary, description)

class PadIssueReporter(object):
    def __init__(self, mqtt_fqdn, mqtt_port):
        self.mqtt_client = ItriMqttClient(mqtt_fqdn, mqtt_port)


    def run(self):
        """Send out aggregated info to backend server every second."""
        rospy.init_node("PadIssueReporter")
        rospy.logwarn("Init PadIssueReporter")
        rate = rospy.Rate(1)
        self.mqtt_client.subscribe(_MQTT_REQ_REPORT_ISSUE_TOPIC, _post_issue)
        while not rospy.is_shutdown():
            rate.sleep()
