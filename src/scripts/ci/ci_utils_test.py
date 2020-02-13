#!/usr/bin/env python
import unittest
import logging
import ci_utils


class CIUtilTest(unittest.TestCase):
    def test_get_repo_path(self):
        repo = ci_utils.get_repo_path()
        logging.info("repo dir: %s", repo)
        self.assertTrue(repo.endswith("itriadv"))

    def test_get_compile_command(self):
        cmd = ci_utils.get_compile_command("src/sensing/itri_parknet/src/parknet_node_impl.cpp")
        self.assertTrue(isinstance(cmd, list))
        self.assertTrue(len(cmd) > 0)


if __name__ == "__main__":
    unittest.main()
