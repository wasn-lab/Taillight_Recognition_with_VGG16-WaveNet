import unittest
import os
import io
import subprocess
from sb_rosbag_sender import SBRosbagSender

class SBRosbagSenderTest(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        ini = os.path.join(cur_dir, "sb_rosbag_sender.ini")
        self.rosbag_dir = os.path.join(cur_dir, "bag_dir")
        self.sender = SBRosbagSender(ini, self.rosbag_dir)
        if not os.path.isdir(self.rosbag_dir):
            os.makedirs(self.rosbag_dir)
        self.bag_names = ["auto_record_2020-09-08-06-24-34_18.bag",
                          "auto_record_2020-10-06-16-39-20_77.bag",
                          "auto_record_2020-10-06-16-39-05_76.bag"]
        for bag in self.bag_names:
            subprocess.call(["touch", os.path.join(self.rosbag_dir, bag)])

    def tearDown(self):
        subprocess.call(["rm", "-r", self.rosbag_dir])

    def test_get_bag_files(self):
        ret = self.sender.get_bag_files()
        self.assertEqual(len(ret), 3)
        self.assertTrue(ret[0][0] != '/')

    def test_get_sb_bag_name(self):
        bags = self.sender.get_bag_files()
        sb_fn = self.sender.get_sb_bag_name(bags[0])
        self.assertEqual(sb_fn, "itri_dc5360f91e74_camera_20200908_0624_1.bag")
        sb_fn = self.sender.get_sb_bag_name(bags[1])
        self.assertEqual(sb_fn, "itri_dc5360f91e74_camera_20201006_1639_1.bag")
        sb_fn = self.sender.get_sb_bag_name(bags[2])
        self.assertEqual(sb_fn, "itri_dc5360f91e74_camera_20201006_1639_2.bag")

    def test_generate_lftp_script(self):
        script = self.sender.generate_lftp_script()
        self.assertTrue("sftp://" in script)
        for bag in self.bag_names:
            self.assertTrue(bag in script)
        script2 = self.sender.generate_lftp_script()
        self.assertEqual(script, script2)


if __name__ == "__main__":
    unittest.main()
