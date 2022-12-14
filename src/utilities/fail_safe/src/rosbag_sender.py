#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
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
    if not rospy.get_param("/fail_safe/should_send_bags", True):
        rospy.logwarn("Do not send bag due to /fail_safe/should_send_bags is False")
        return 0

    shell_cmd = ["lftp", "-f", lftp_script_file]
    rospy.logwarn(" ".join(shell_cmd))
    if debug_mode:
        logging.warn("In debug mode, do not actually send bag files.")
        return 0

    try:
        ret = subprocess.check_call(shell_cmd)
    except subprocess.CalledProcessError:
        rospy.logwarn("Fail to upload bag file. lftp_script path: %s", lftp_script_file)
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
        rospy.logwarn("rosbag backup dir: {}".format(self.rosbag_backup_dir))
        rospy.logwarn("backend server fqdn: {}, user_name: {}".format(self.fqdn, self.user_name))

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.send_if_having_bags()
            self.heartbeat_publisher.publish(Empty())
            rate.sleep()

    def send_if_having_bags(self):
        bags = self.get_unsent_rosbag_filenames()
        if not bags:
            rospy.logwarn("{}: No bags to upload".format(datetime.datetime.now()))
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

    def send_bags(self, bags):
        bags.sort()

        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            if bag_base_name in self.notified_bags:
                continue
            if not self.debug_mode:
                rospy.logwarn("notify backend: %s is ready to be uploaded", bag)
                jret = notify_backend_with_new_bag(bag_base_name)
                rospy.logwarn(jret)
                self.notified_bags[bag_base_name] = True

        should_notify_backend = (
            rospy.get_param("/fail_safe/should_notify_backend", True)
            and not self.debug_mode)
        for bag in bags:
            _, bag_base_name = os.path.split(bag)
            lftp_script_filename = self._generate_lftp_script(bag)
            ret = _send_bag_by_lftp(lftp_script_filename, debug_mode=self.debug_mode)
            if ret == 0 and should_notify_backend:
                rospy.logwarn("notify backend: %s has been uploaded successfuly", bag)
                jret = notify_backend_with_uploaded_bag(bag_base_name)
                rospy.logwarn(jret)
            if os.path.isfile(lftp_script_filename):
                os.unlink(lftp_script_filename)

    def _generate_lftp_script(self, bag):
        ftp_cmds = [
            u"set ssl:verify-certificate no",
            u"set net:limit-total-rate 0:{}".format(self.upload_rate),
            u"set sftp:auto-confirm yes",
            u"open -p {} -u {},{} {}".format(self.port, self.user_name, self.password, self.fqdn),
        ]
        ymd = get_bag_yymmdd(bag)  # backup dir name in backend
        dir_name = u"/fail_safe/{}/{}".format(self.license_plate_number, ymd)
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
