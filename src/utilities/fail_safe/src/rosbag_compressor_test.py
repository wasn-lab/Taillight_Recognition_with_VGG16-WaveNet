# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import configparser
import os
import io
from rosbag_compressor import compress_bag, decompress_bag, should_compress_bag

class RosbagCompressorTest(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(cur_dir, "data")
        bag_name = "auto_record_2020-10-06-16-20-50_3.bag"
        self.bag_gz = os.path.join(data_dir, bag_name + ".gz")
        self.bag = os.path.join(data_dir, bag_name)
        decompress_bag(self.bag_gz)  # now we have |bag_name|

    def tearDown(self):
        ratio = compress_bag(self.bag)

    def test_compress_bag(self):
        ratio = compress_bag(self.bag)
        print("compression ratio: {}".format(ratio))
        self.assertTrue(ratio > 0 and ratio <= 1)

        ratio = compress_bag(self.bag_gz)
        self.assertEqual(ratio, -1)

    def test_should_compress_bag(self):
        root = "/home/camera/rosbag_files/backup"
        filename = "auto_record_2021-04-27-00-23-30_16.bag"
        self.assertTrue(should_compress_bag(root, filename))

        root = "/home/camera/rosbag_files/route_02_auto/backup"
        filename = "auto_record_2021-04-27-00-23-30_16.bag"
        self.assertTrue(should_compress_bag(root, filename))

        root = "/home/camera/rosbag_files/backup"
        filename = "auto_record_2021-04-27-00-23-30_16.bag.gz"
        self.assertFalse(should_compress_bag(root, filename))

        root = "/home/camera/rosbag_files/tmp"
        filename = "auto_record_2021-04-27-00-23-30_16.bag"
        self.assertFalse(should_compress_bag(root, filename))

if __name__ == "__main__":
    unittest.main()
