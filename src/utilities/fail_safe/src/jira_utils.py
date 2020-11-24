"""
API for interacting with jira.
"""
import logging
import json
import requests
from requests.auth import HTTPBasicAuth
# from sb_param_utils import get_license_plate_number

JIRA_BASE_URL = "https://jira.itriadv.co"
JIRA_USER = "icl_u300"
JIRA_PASSWORD = "itriu300"

PROJECT_ID_P_S3 = "10405"
PROJECT_ID_SCM = "10400"
ISSUE_TYPE_ID_BUG = "10100"
ISSUE_TYPE_ID_TASK = "10002"

ALLOWED_PROJECT_IDS = set([PROJECT_ID_P_S3, PROJECT_ID_SCM])
ALLOWED_ISSUE_TYPE_IDS = set([ISSUE_TYPE_ID_BUG, ISSUE_TYPE_ID_TASK])

def get_issue_contents(issue_id):
    """
    issue_id (str) -- Issue id like SCM-32

    Return response from the server. Useful fields: text, status_code,
    """
    url = "{}/rest/api/latest/issue/{}".format(JIRA_BASE_URL, issue_id)
    req_headers = {"Content-Type": "application/json"}
    return requests.get(url, headers=req_headers,
                        auth=HTTPBasicAuth(JIRA_USER, JIRA_PASSWORD),
                        verify=False)


def generate_issue_contents(project_id, summary, description, issue_type_id):
    """
    Generate issue contents that jira server expects
    """
    if project_id not in ALLOWED_PROJECT_IDS:
        logging.error("unexpted project id: %s", project_id)
        return {}
    if issue_type_id not in ALLOWED_ISSUE_TYPE_IDS:
        logging.error("unexpted issue type id: %s", issue_type_id)
        return {}

    if not summary:
        logging.warn("Empty summary when creating issue.")
    if not description:
        logging.warn("Empty description when creating issue.")

    return {
        "fields": {
            "project": {"id": project_id},
            "summary": summary,
            "description": description,
            "labels": ["FieldTest"],
            "issuetype": {"id": issue_type_id}}}

def post_issue(project_id, summary, description, issue_type_id):
    """
    Post issue to jira server.

    Return 0 if success, 1 otherwise.
    """
    contents = generate_issue_contents(
        project_id, summary, description, issue_type_id)
    if not contents:
        logging.warn("Issue is not posted")
        return 1
    url = "{}/rest/api/latest/issue/".format(JIRA_BASE_URL)
    req_headers = {"Content-Type": "application/json"}
    resp = requests.post(
        url, data=json.dumps(contents),
        headers=req_headers,
        auth=HTTPBasicAuth(JIRA_USER, JIRA_PASSWORD),
        verify=False)
    if resp.status_code / 100 != 2:
        logging.warning("Fail to post issue, server response: %s", resp.text)
        return 1
    return 0
