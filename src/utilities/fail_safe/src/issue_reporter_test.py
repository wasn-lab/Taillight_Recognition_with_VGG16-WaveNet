# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import datetime
from issue_reporter import IssueReporter, generate_issue_description
from jira_utils import PROJECT_ID_SCM, ISSUE_TYPE_ID_TASK
from status_level import OK, WARN


class IssueReporterTest(unittest.TestCase):
    def test_post_issue(self):
        reporter = IssueReporter()
        reporter.set_project_id(PROJECT_ID_SCM)
        reporter.set_issue_type_id(ISSUE_TYPE_ID_TASK)
        summary = "[auto report] test issue reporter"
        description = "This is description"
        num_succ = 0
        for _ in range(10):
            num_succ += reporter.post_issue(summary, description, dry_run=True)
        self.assertEqual(num_succ, 1)

    def test_generate_issue_description(self):
        status_str = "Misbehaving modules: ACC AEB XByWire CAN"
        timestamp = 1608530369.999671  # (2020, 12, 21, 13, 59, 29, 999671)
        desc = generate_issue_description(OK, status_str, timestamp)
        self.assertEqual(desc, "")

        desc = generate_issue_description(WARN, status_str, timestamp)
        self.assertTrue(len(desc) > 0)
        url = (u"https://service.itriadv.co:8743/ADV/EventPlayback?"
               u"plate=試0002&startDt=2020-12-21 13:59&endDt=2020-12-21 14:00")
        self.assertTrue(url in desc)

    def test_generate_issue_description_timestamp_mot(self):
        status_str = "Misbehaving modules: ACC AEB XByWire CAN"
        timestamp = 1608530369999 # (2020, 12, 21, 13, 59, 29, 999671)
        desc = generate_issue_description(OK, status_str, timestamp)
        self.assertEqual(desc, "")

        desc = generate_issue_description(WARN, status_str, timestamp)
        self.assertTrue(len(desc) > 0)
        url = (u"https://service.itriadv.co:8743/ADV/EventPlayback?"
               u"plate=試0002&startDt=2020-12-21 13:59&endDt=2020-12-21 14:00")
        self.assertTrue(url in desc)


if __name__ == "__main__":
    unittest.main()
