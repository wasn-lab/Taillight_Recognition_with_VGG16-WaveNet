import unittest
import time
import configparser
import os
from rosbag_utils import get_bag_yymmdd


class RosbagUtilsTest(unittest.TestCase):
    def test_get_bag_yymmdd(self):
        bag = "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(get_bag_yymmdd(bag), "20201006")
        bag = "auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(get_bag_yymmdd(bag), "20201006")


if __name__ == "__main__":
    unittest.main()
