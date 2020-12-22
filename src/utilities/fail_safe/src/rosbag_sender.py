# -*- encoding: utf-8 -*-
"""
Send backup rosbag files to backend.
"""
from __future__ import print_function
import time
import datetime
import os
import io
import subprocess
import rospy
from std_msgs.msg import Empty
from sb_param_utils import get_license_plate_number
from rosbag_utils import get_bag_yymmdd
from vk221_3 import notify_backend_with_new_bag
from vk221_4 import notify_backend_with_uploaded_bag

_BACKUP_ROSBAG_LFTP_SCRIPT = "/tmp/backup_rosbag_lftp_script.txt"


def _get_stamp_filename(fullpath):
    """.stamp file indicate if we already send the corresponding bag."""
    return fullpath[:-4] + ".stamp"


def _send_bag_by_lftp(lftp_script_file):
    shell_cmd = ["lftp", "-f", lftp_script_file]
    print(" ".join(shell_cmd))
    try:
        ret = subprocess.check_call(shell_cmd)
    except subprocess.CalledProcessError:
        print("Fail to upload bag file. lftp_script contents:")
        with io.open(lftp_script_file) as _fp:
            print(_fp.read())
        ret = 1
    return ret


class RosbagSender(object):
    def __init__(self, fqdn, port, user_name, password, rosbag_backup_dir, upload_rate):
        """
        Currently we use FTP protocol to send rosbags
        """
        rospy.init_node("RosbagSender")
        rospy.logwarn("Init RosbagSender")
        self.heartbeat_publisher = rospy.Publisher(
            "/fail_safe/rosbag_sender/heartbeat", Empty, queue_size=10)
        self.fqdn = fqdn
        self.port = port
        self.license_plate_number = get_license_plate_number()
        self.proc = None
        self.user_name = user_name
        self.password = password
        self.upload_rate = upload_rate
        self.notified_bags = {}
        if not rosbag_backup_dir:
            self.rosbag_backup_dir = os.path.join(
                os.environ["HOME"], "rosbag_files", "backup")
        else:
            self.rosbag_backup_dir = rosbag_backup_dir
        print("rosbag backup dir: {}".format(self.rosbag_backup_dir))
        print("backend server fqdn: {}, user_name: {}".format(self.fqdn, self.user_name))

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.send_if_having_bags()
            self.heartbeat_publisher.publish(Empty())
            rate.sleep()

    def send_if_having_bags(self):
        bags = self.get_unsent_rosbag_filenames()
        if not bags:
            print("{}: No bags to upload".format(datetime.datetime.now()))
            return
        self.send_bags(bags)

    def send_bags(self, bags):
        bags.sort()

        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            if bag_base_name in self.notified_bags:
                continue
            print("notify backend: {} is ready to be uploaded".format(bag))
            jret = notify_backend_with_new_bag(bag_base_name)
            print(jret)
            self.notified_bags[bag_base_name] = True

        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            lftp_script_filename = self._generate_lftp_script(bag)
            ret = _send_bag_by_lftp(lftp_script_filename)
            if ret == 0:
                print("notify backend: {} has been uploaded successfuly".format(bag))
                jret = notify_backend_with_uploaded_bag(bag_base_name)
                print(jret)

    def _generate_lftp_script(self, bag):
        ftp_cmds = [
            u"set ssl:verify-certificate no",
            u"set net:limit-total-rate 0:{}".format(self.upload_rate),
            u"open -p {} -u {},{} {}".format(self.port, self.user_name, self.password, self.fqdn),
        ]
        ymd = get_bag_yymmdd(bag)  # backup dir name in backend
        dir_name = u"/Share/ADV/Rosbag/fail_safe/{}/{}".format(self.license_plate_number, ymd)
        ftp_cmds += [
            u"mkdir -p {}".format(dir_name),
            u"cd {}".format(dir_name),
            u"put -c {}".format(bag),
            u"!touch {}".format(_get_stamp_filename(bag)),
        ]
        ftp_cmds += [u"bye"]

        if os.path.isfile(_BACKUP_ROSBAG_LFTP_SCRIPT):
            os.unlink(_BACKUP_ROSBAG_LFTP_SCRIPT)

        with io.open(_BACKUP_ROSBAG_LFTP_SCRIPT, "w", encoding="utf-8") as _fp:
            _fp.write(u"\n".join(ftp_cmds))
            _fp.write(u"\n")
        return _BACKUP_ROSBAG_LFTP_SCRIPT

    def get_unsent_rosbag_filenames(self):
        """Return a list of bag files that are not sent back"""
        ret = []
        if not os.path.isdir(self.rosbag_backup_dir):
            rospy.logwarn("No rosbag backup dir: %s", self.rosbag_backup_dir)
            return ret

        files = os.listdir(self.rosbag_backup_dir)
        for fname in files:
            if not fname.endswith(".bag"):
                continue
            fullpath = os.path.join(self.rosbag_backup_dir, fname)
            if not os.path.isfile(fullpath):
                continue
            if not fname.endswith(".bag"):
                continue
            stamp_fname = _get_stamp_filename(fullpath)
            if not os.path.isfile(stamp_fname):
                ret.append(fullpath)
        return ret
