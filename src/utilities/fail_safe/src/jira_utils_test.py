import unittest
# import pprint
from jira_utils import (
    get_issue_contents, generate_issue_contents, post_issue,
    PROJECT_ID_P_S3, ISSUE_TYPE_ID_BUG, PROJECT_ID_SCM)


class JiraUtilsTest(unittest.TestCase):
    def test_get_issue_contents(self):
        issue_id = "S3-229"
        resp = get_issue_contents(issue_id)
        # jdata = resp.json()
        # pprint.pprint(jdata)
        # self.assertEqual(jdata["fields"]["labels"], ["FieldTest"])
        self.assertEqual(resp.status_code, 200)

    def test_generate_issue_contents(self):
        ret = generate_issue_contents(
            0, "summary", "description", ISSUE_TYPE_ID_BUG)
        self.assertEqual(ret, {})

        ret = generate_issue_contents(
            PROJECT_ID_P_S3, "summary", "description", 0)
        self.assertEqual(ret, {})

        ret = generate_issue_contents(
            PROJECT_ID_P_S3, "my summary", "my lengthy description",
            ISSUE_TYPE_ID_BUG)
        self.assertEqual(ret["fields"]["summary"], "my summary")
        self.assertEqual(ret["fields"]["description"], "my lengthy description")
        self.assertEqual(ret["fields"]["project"]["id"], PROJECT_ID_P_S3)
        self.assertEqual(ret["fields"]["issuetype"]["id"], ISSUE_TYPE_ID_BUG)

    @unittest.skip("Manual test")
    def test_post_issue(self):
        ret = post_issue(
            PROJECT_ID_P_S3,
            "[auto report] Test fail safe auto issue report",
            "Description of fail safe auto issue report",
            ISSUE_TYPE_ID_BUG)
        self.assertEqual(ret, 0)

if __name__ == "__main__":
    unittest.main()
