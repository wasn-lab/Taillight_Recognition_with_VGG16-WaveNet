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

    @unittest.skip("Dependent on local env")
    def test_1(self):
        bags = self.sender.get_unsent_rosbag_filenames()
        self.assertTrue(len(bags) > 0)

    def test__generate_lftp_script(self):
        bag = "auto_record_2020-10-06-16-24-34_18.bag"
        filename = self.sender._generate_lftp_script(bag)
        with io.open(filename) as _fp:
            contents = _fp.read()
        print(contents)
        self.assertTrue(bag in contents)

    def test_send_bags(self):
        bags = [
            "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-24-20_17.bag",
            "/media/chtseng/Sandisk/rosbag_files/backup/auto_record_2020-10-06-16-24-34_18.bag"]
        self.sender.send_bags(bags)

    @unittest.skip("Manually enabled test item")
    def test_run(self):
        self.sender.run()


if __name__ == "__main__":
    unittest.main()
