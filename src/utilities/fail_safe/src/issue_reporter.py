import time
import logging
import datetime
from status_level import STATUS_CODE_TO_STR, OK
from jira_utils import (post_issue, PROJECT_ID_P_S3, ISSUE_TYPE_ID_BUG)


def generate_issue_description(status_code, status_str):
    if status_code == OK:
        logging.warn("Do not generate description for status OK")
        return ""
    return """
status code: {}

status str: {}

issue is reported at timestamp: {}

Please login to https://service.itriadv.co:8743/
to retrieve related bag files.

- User name: U200
- User password: u200u200u200
    """.format(STATUS_CODE_TO_STR[status_code], status_str, datetime.datetime.now())


class IssueReporter():
    def __init__(self):
        self.last_posts = {}  # "summary": time (float)
        self.min_post_time_interval = 60  # Don't post same issue within 60s
        self.project_id = PROJECT_ID_P_S3
        self.issue_type_id = ISSUE_TYPE_ID_BUG

    def set_project_id(self, project_id):
        """set project id"""
        self.project_id = project_id

    def set_issue_type_id(self, issue_type_id):
        """set issue type"""
        self.issue_type_id = issue_type_id

    def post_issue(self, summary, description, dry_run=False):
        """
        Returns 1 if actually post an issue. 0 otherwise.
        """
        last_post = self.last_posts.get(summary, 0)
        now = time.time()
        self.last_posts[summary] = now
        if now - last_post <= self.min_post_time_interval:
            logging.warn("%s: Does not post repeated issue", summary)
            return 0

        if dry_run:
            logging.warn("%s: Dry run mode. Do not post issue to jira", summary)
        else:
            logging.warn("%s: Post issue to jira", summary)
            post_issue(self.project_id, summary, description, self.issue_type_id)

        return 1
