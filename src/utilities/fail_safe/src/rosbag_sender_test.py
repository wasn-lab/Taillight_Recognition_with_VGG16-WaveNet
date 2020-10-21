import unittest
import time
import configparser
import os
from rosbag_sender import RosbagSender, _get_bag_ymd

class RosbagSenderTest(unittest.TestCase):
    def setUp(self):
        cfg = configparser.ConfigParser()
        cur_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(cur_path, "rosbag_sender.ini")
        cfg.read(cfg_file)
        self.sender = RosbagSender(cfg["ftp"]["fqdn"], cfg["ftp"]["port"],
            "U300", "/media/chtseng/Sandisk/rosbag_files/backup")

    @unittest.skipIf(True, "")
    def test_1(self):
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertTrue(len(bags) > 0)

    def test_2(self):
        bag = "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(_get_bag_ymd(bag), "20201006")

    def test_run(self):
        self.sender.run();


if __name__ == "__main__":
    unittest.main()
