import unittest
import time
import configparser
import os
from rosbag_sender import RosbagSender

class RosbagSenderTest(unittest.TestCase):
    def setUp(self):
        cfg = configparser.ConfigParser()
        cur_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(cur_path, "rosbag_sender.ini")
        cfg.read(cfg_file)
        self.sender = RosbagSender(cfg["ftp"]["fqdn"], cfg["ftp"]["port"],
            "U300", None, "/media/chtseng/Sandisk/rosbag_files/backup")

    @unittest.skipIf(True, "")
    def test_1(self):
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertTrue(len(bags) > 0)

    def test_run(self):
        self.sender.run();


if __name__ == "__main__":
    unittest.main()
