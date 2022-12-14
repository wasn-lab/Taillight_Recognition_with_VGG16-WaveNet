# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import datetime
import configparser
import os
import io
import subprocess
from rosbag_sender import RosbagSender, _should_delete_bag


class RosbagSenderTest(unittest.TestCase):
    def setUp(self):
        cfg = configparser.ConfigParser()
        cur_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(cur_path, "rosbag_sender.ini")
        cfg.read(cfg_file)
        self.sender = RosbagSender(
            cfg["ftp"]["fqdn"],
            cfg["ftp"]["port"],
            cfg["ftp"]["user_name"],
            cfg["ftp"]["password"],
            cfg["rosbag"]["backup_dir"],
            cfg["ftp"]["upload_rate"])
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(cur_dir, "data")
        self.sender.set_rosbag_backup_dir(self.data_dir)

        self.sender.set_debug_mode(True)
        rel_bag = "auto_record_2020-10-06-16-20-50_3.bag"
        self.bag = os.path.join(cur_dir, "data", rel_bag)
        self.bag_gz = self.bag + ".gz"

    def test__should_delete_bag(self):
        bag = "/home/nvidia/rosbag_files/backup/auto_record_2020-10-06-16-26-50_27.bag"
        self.assertTrue(_should_delete_bag(bag))
        bag_dt = datetime.datetime(year=2020, month=10, day=6, hour=16, minute=26, second=50)

        self.assertFalse(
            _should_delete_bag(bag, bag_dt + datetime.timedelta(days=0)))
        self.assertFalse(
            _should_delete_bag(bag, bag_dt + datetime.timedelta(days=1)))
        self.assertFalse(
            _should_delete_bag(bag, bag_dt + datetime.timedelta(days=2)))
        self.assertFalse(
            _should_delete_bag(bag, bag_dt + datetime.timedelta(days=3)))
        self.assertTrue(
            _should_delete_bag(bag, bag_dt + datetime.timedelta(days=4)))

    def test_get_unsent_rosbag_filenames(self):
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertEqual(bags, [self.bag_gz])

        # When .bag and .bag.gz both exist, we should not send it as it is compressing.
        cmd = ["touch", self.bag]
        subprocess.check_call(cmd)
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertEqual(bags, [])
        os.unlink(self.bag)

    def test__generate_lftp_script(self):
        bag_gz = "auto_record_2020-10-06-16-24-34_18.bag.gz"
        filename = self.sender._generate_lftp_script(bag_gz)
        with io.open(filename) as _fp:
            contents = _fp.read()
        self.assertTrue(bag_gz in contents)

    @unittest.skip("Manually enabled test item")
    def test_send_bags(self):
        self.sender.set_rosbag_backup_dir("/media/chtseng/Sandisk/20201228/full_run")
        bags = self.sender.get_unsent_rosbag_filenames()
        self.sender.send_bags(bags)

    @unittest.skip("Manually enabled test item")
    def test_run(self):
        self.sender.run()


if __name__ == "__main__":
    unittest.main()
