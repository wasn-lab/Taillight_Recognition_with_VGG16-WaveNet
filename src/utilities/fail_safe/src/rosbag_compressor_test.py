import unittest
import configparser
import os
import io
from rosbag_compressor import compress_bag, decompress_bag

class RosbagCompressorTest(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(cur_dir, "data")
        bag_name = "auto_record_2020-10-06-16-20-50_3.bag"
        self.bag_gz = os.path.join(data_dir, bag_name + ".gz")
        self.bag = os.path.join(data_dir, bag_name)
        decompress_bag(self.bag_gz)  # now we have |bag_name|

    def testDown(self):
        ratio = compress_bag(self.bag)

    def test_compress_bag(self):
        ratio = compress_bag(self.bag)
        print("compression ratio: {}".format(ratio))
        self.assertTrue(ratio > 0 and ratio <= 1)

        ratio = compress_bag(self.bag_gz)
        self.assertEqual(ratio, -1)

if __name__ == "__main__":
    unittest.main()
