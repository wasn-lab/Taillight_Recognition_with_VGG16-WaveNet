# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import time
import logging
import datetime
from status_level import STATUS_CODE_TO_STR, OK
from jira_utils import (post_issue, PROJECT_ID_P_S3, ISSUE_TYPE_ID_BUG)
from sb_param_utils import get_license_plate_number


def generate_crash_description(exc_str, module="Fail-safe"):
    dt = datetime.datetime.fromtimestamp(time.time())
    plate = get_license_plate_number()
    return u"""
{} crashed at timestamp: {}
License plate: {}
{}
""".format(module, dt, plate, exc_str)


def generate_issue_description(status_code, status_str, timestamp):
    if status_code == OK:
        logging.warn("Do not generate description for status OK")
        return ""
    # timestamp is expected to be of length 10
    ts_len = len(str(int(timestamp)))
    if ts_len == 13:
        # South bridge format.
        timestamp = timestamp / 1000
    dt = datetime.datetime.fromtimestamp(timestamp)
    plate = get_license_plate_number()
    start_dt = "{}-{}-{} {}:{}".format(
        dt.year, dt.month, dt.day, dt.hour, dt.minute)
    minute_after_dt = dt + datetime.timedelta(minutes=1)
    end_dt = "{}-{}-{} {:02d}:{:02d}".format(
        minute_after_dt.year, minute_after_dt.month, minute_after_dt.day,
        minute_after_dt.hour, minute_after_dt.minute)

    url = (u"https://service.itriadv.co:8743/ADV/EventPlayback?plate={}&"
           u"startDt={}&endDt={}").format(plate, start_dt, end_dt)

    return u"""
status code: {}

status str: {}

issue is reported at timestamp: {}

Please use the url
  {}
to retrieve related bag files.

- User name: u200
- User password: please ask your colleague
    """.format(STATUS_CODE_TO_STR[status_code], status_str, dt, url)


class IssueReporter():
    def __init__(self):
        self.lastest_issue_post_time = 0
        self.min_post_time_interval = 60  # Don't post same issue within 60s
        self.project_id = PROJECT_ID_P_S3
        self.issue_type_id = ISSUE_TYPE_ID_BUG

    def set_project_id(self, project_id):
        """set project id"""
        self.project_id = project_id

    def set_issue_type_id(self, issue_type_id):
        """set issue type"""
        self.issue_type_id = issue_type_id

    def _is_repeated_issue(self):
        now = time.time()
        prev_post_time = self.lastest_issue_post_time
        self.lastest_issue_post_time = now
        return bool(now - prev_post_time <= self.min_post_time_interval)

    def post_issue(self, summary, description, dry_run=False):
        """
        Returns 1 if actually post an issue. 0 otherwise.
        """
        if self._is_repeated_issue():
            logging.warn("%s: Does not post repeated issue", summary)
            return 0

        if dry_run:
            logging.warn("%s: Dry run mode. Do not post issue to jira", summary)
        else:
            logging.warn("%s: Post issue to jira", summary)
            post_issue(self.project_id, summary, description, self.issue_type_id)

        return 1
