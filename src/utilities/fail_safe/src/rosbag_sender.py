# -*- encoding: utf-8 -*-
"""
Send backup rosbag files to backend.
"""
from __future__ import print_function
# import time
import datetime
import os
import io
import subprocess
import logging
import rospy
from std_msgs.msg import Empty
from sb_param_utils import get_license_plate_number
from rosbag_utils import get_bag_yymmdd, get_bag_timestamp
from vk221_3 import notify_backend_with_new_bag
from vk221_4 import notify_backend_with_uploaded_bag


def _should_delete_bag(bag_fullpath, current_dt=None):
    """
    Return True if the datetime in |bag_fullpath| is <= |current_dt| - 3 days.
    Return False otherwise.
    """
    bag_dt = get_bag_timestamp(bag_fullpath)
    if current_dt is None:
        current_dt = datetime.datetime.now()
    delta = current_dt - bag_dt
    return bool(delta.total_seconds() > 259200)  # 259200 = 3*24*60*60


def _get_stamp_filename(fullpath):
    """.stamp file indicate if we already send the corresponding bag."""
    return fullpath + ".stamp"


def _send_bag_by_lftp(lftp_script_file, debug_mode=False):
    shell_cmd = ["lftp", "-f", lftp_script_file]
    print(" ".join(shell_cmd))
    if debug_mode:
        logging.warn("In debug mode, do not actually send bag files.")
        return 0

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
        self.debug_mode = False
        self.set_rosbag_backup_dir(rosbag_backup_dir)
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

    def set_debug_mode(self, mode):
        self.debug_mode = mode

    def set_rosbag_backup_dir(self, rosbag_backup_dir):
        if not rosbag_backup_dir:
            self.rosbag_backup_dir = os.path.join(
                os.environ["HOME"], "rosbag_files", "backup")
        else:
            self.rosbag_backup_dir = rosbag_backup_dir

    def _delete_old_bags_if_necessary(self, bags):
        for bag in bags:
            if not _should_delete_bag(bag):
                continue

            for filename in [bag, _get_stamp_filename(bag)]:
                if not os.path.isfile(filename):
                    continue
                rospy.logwarn("rm %s", filename)
                if self.debug_mode:
                    rospy.logwarn("Debug mode: do not actually rm %s", filename)
                else:
                    os.unlink(filename)

    def send_bags(self, bags):
        bags.sort()
        self._delete_old_bags_if_necessary(bags)

        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            if bag_base_name in self.notified_bags:
                continue
            if not self.debug_mode:
                print("notify backend: {} is ready to be uploaded".format(bag))
                jret = notify_backend_with_new_bag(bag_base_name)
                print(jret)
                self.notified_bags[bag_base_name] = True

        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            lftp_script_filename = self._generate_lftp_script(bag)
            ret = _send_bag_by_lftp(lftp_script_filename, self.debug_mode)
            if ret == 0 and not self.debug_mode:
                print("notify backend: {} has been uploaded successfuly".format(bag))
                jret = notify_backend_with_uploaded_bag(bag_base_name)
                print(jret)
            if os.path.isfile(lftp_script_filename):
                os.unlink(lftp_script_filename)

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

        script_file = os.path.join("/tmp", bag + ".lftprc")
        if os.path.isfile(script_file):
            os.unlink(script_file)

        with io.open(script_file, "w", encoding="utf-8") as _fp:
            _fp.write(u"\n".join(ftp_cmds))
            _fp.write(u"\n")
        return script_file

    def get_unsent_rosbag_filenames(self):
        """Return a list of bag files that are not sent back"""
        ret = []
        if not os.path.isdir(self.rosbag_backup_dir):
            rospy.logwarn("No rosbag backup dir: %s", self.rosbag_backup_dir)
            return ret

        for root, _dirs, files in os.walk(self.rosbag_backup_dir):
            for fname in files:
                if not fname.endswith(".bag.gz"):
                    continue
                fullpath = os.path.join(root, fname)
                bag = fullpath[:-3]
                if os.path.isfile(bag):
                    # still compressing, skip this file
                    continue
                stamp_fname = _get_stamp_filename(fullpath)
                if not os.path.isfile(stamp_fname):
                    ret.append(fullpath)
        return ret
