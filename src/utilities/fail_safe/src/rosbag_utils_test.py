# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
from rosbag_utils import get_bag_yymmdd, get_bag_timestamp, get_bag_timestamp_in_dict


class RosbagUtilsTest(unittest.TestCase):
    def test_get_bag_yymmdd(self):
        bag = "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(get_bag_yymmdd(bag), "20201006")
        bag = "auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(get_bag_yymmdd(bag), "20201006")

    def test_get_bag_timestamp(self):
        bag = "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        ret = get_bag_timestamp(bag)
        self.assertEqual(ret.year, 2020)
        self.assertEqual(ret.month, 10)
        self.assertEqual(ret.day, 6)
        self.assertEqual(ret.hour, 16)
        self.assertEqual(ret.minute, 26)
        self.assertEqual(ret.second, 50)

    def test_get_bag_timestamp_in_dict(self):
        bag = "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        expect = {
            "year": "2020",
            "month": "10",
            "day": "06",
            "hour": "16",
            "minute": "26",
            "second": "50"}
        self.assertEqual(get_bag_timestamp_in_dict(bag), expect)
        bag = "auto_record_2020-10-06-16-26-50_27.bag"
        self.assertEqual(get_bag_timestamp_in_dict(bag), expect)

if __name__ == "__main__":
    unittest.main()
