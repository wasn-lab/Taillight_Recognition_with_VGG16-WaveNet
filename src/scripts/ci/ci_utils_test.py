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

    def test_get_compile_command(self):
        cpp = "src/sensing/itri_drivenet/drivenet/src/drivenet_120_1_b1.cpp"
        args = ci_utils.get_compile_args(cpp)
        self.assertTrue(isinstance(args, list))
        self.assertTrue(len(args) > 0)
        self.assertTrue("clang" not in args[0])
        self.assertTrue("c++" not in args[0])
        self.assertTrue("ccache" not in args[0])
        self.assertTrue("-o" not in args)
        self.assertTrue("-c" not in args)
        for arg in args:
            self.assertTrue(not arg.endswith(".o"))
            self.assertTrue(not arg.endswith(cpp))


if __name__ == "__main__":
    unittest.main()
