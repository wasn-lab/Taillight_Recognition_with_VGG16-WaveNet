import unittest
import configparser
import os
import io
from rosbag_sender import RosbagSender

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
        self.sender.set_rosbag_backup_dir(os.path.join(cur_dir, "data"))
        self.sender.set_debug_mode(True)

    def test_1(self):
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertEqual(len(bags), 1)
        self.assertTrue("auto_record_2020-10-06-16-20-50_3.bag.gz" in bags[0])

    def test__generate_lftp_script(self):
        bag_gz = "auto_record_2020-10-06-16-24-34_18.bag.gz"
        filename = self.sender._generate_lftp_script(bag_gz)
        with io.open(filename) as _fp:
            contents = _fp.read()
        self.assertTrue(bag_gz in contents)

    def test_send_bags(self):
        self.sender.set_rosbag_backup_dir("/media/chtseng/Sandisk/20201228/full_run")
        bags = self.sender.get_unsent_rosbag_filenames()
        self.sender.send_bags(bags)

    @unittest.skip("Manually enabled test item")
    def test_run(self):
        self.sender.run()


if __name__ == "__main__":
    unittest.main()
