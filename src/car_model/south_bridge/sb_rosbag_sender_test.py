#!/usr/bin/env python
import unittest
from sb_rosbag_sender import get_default_bag_dir, get_bag_filenames, convert_to_sb_bag_name, get_sb_config, get_bag_yymmdd, gen_sftp_cmd

class SBRosbagSenderTest(unittest.TestCase):
    def test_get_bag_filenames(self):
        bag_dir = get_default_bag_dir()
        actual = get_bag_filenames(bag_dir)
        #print(actual)
        self.assertTrue(len(actual) > 0)

    def test_get_sb_config(self):
        actual = get_sb_config()
        self.assertEqual(actual["sftp_ip"], "140.110.141.115")

    def test_get_bag_yymmdd(self):
        fullpath = "/home/nvidia/rosbag_files/tmp/auto_record_2020-10-06-16-44-19_97.bag"
        self.assertEqual(get_bag_yymmdd(fullpath), "20201006")
        fullpath = "auto_record_2020-10-06-16-44-19_97.bag"
        self.assertEqual(get_bag_yymmdd(fullpath), "20201006")

    def test_convert_to_sb_bag_name(self):
        fullpath = "/home/nvidia/rosbag_files/tmp/auto_record_2020-10-06-16-44-19_97.bag"
        actual = convert_to_sb_bag_name(fullpath, 177)
        expect = "itri_dc5360f91e74_camera_20201006_1644_177.bag"
        self.assertEqual(actual, expect)

    def test_gen_sftp_cmd(self):
        bag = "/home/nvidia/rosbag_files/tmp/auto_record_2020-10-06-16-44-19_97.bag"
        actual = gen_sftp_cmd(bag, 123)
        print(actual)

if __name__ == "__main__":
    unittest.main()
