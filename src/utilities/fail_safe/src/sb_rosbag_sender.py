# -*- encoding: utf-8 -*-
"""
Generate lftp scripts for uploading rosbag files.
"""
from __future__ import print_function
import configparser
import os
import io
import re
import rospy
from sb_param_utils import get_license_plate_number, get_company_name, get_vid

_BAG_RGX = re.compile(
    r".+_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})_[\d]+.bag")


def get_bag_yymmdd(bag):
    _path, bag_fn = os.path.split(bag)
    match = _BAG_RGX.search(bag_fn)
    if not match:
        logging.warn("Cannot parse filename: %s", bag)
        return "999999"
    year = match.expand(r"\g<year>")
    month = match.expand(r"\g<month>")
    day = match.expand(r"\g<day>")
    return year + month + day


class SBRosbagSender(object):
    def __init__(self, ini_fn, rosbag_dir):
        """
        Currently we use FTP protocol to send rosbags
        """
        #rospy.init_node("SBRosbagSender")
        #rospy.logwarn("Init SBRosbagSender: ini file: %s, rosbag_dir: %s", ini_fn, rosbag_dir)
        cfg = configparser.ConfigParser()
        cfg.read(ini_fn)

        self.sftp_ip = cfg["south_bridge_sftp"]["sftp_ip"]
        self.port = cfg["south_bridge_sftp"]["port"]
        self.account = cfg["south_bridge_sftp"]["account"]
        self.password = cfg["south_bridge_sftp"]["password"]
        self.company_name = get_company_name()
        self.vid = get_vid()
        self.license_plate_number = get_license_plate_number()
        self.bag_seqs = {}
        self.rosbag_dir = rosbag_dir

    def get_bag_files(self):
        if not os.path.isdir(self.rosbag_dir):
            return []
        ret = []
        for root, _dirs, files in os.walk(self.rosbag_dir):
            for filename in files:
                if ".bag" not in filename:
                    continue
                fullpath = os.path.join(root, filename)
                rel_path = fullpath.replace(self.rosbag_dir, "")
                if rel_path[0] == os.sep:
                    rel_path = rel_path[1:]
                ret.append(rel_path)
        ret.sort()
        return ret

    def _get_sb_bag_basename(self, bag_fullpath):
        _, bag_fn = os.path.split(bag_fullpath)
        match = _BAG_RGX.search(bag_fn)
        if not match:
            print("Cannot parse filename: %s", bag_fullpath)
            return bag_fullpath
        year = match.expand(r"\g<year>")
        month = match.expand(r"\g<month>")
        day = match.expand(r"\g<day>")
        hour = match.expand(r"\g<hour>")
        minute = match.expand(r"\g<minute>")
        # second = match.expand(r"\g<second>")
        fn = (self.company_name + "_" + self.vid + "_camera_" +
              year + month + day + "_" + hour + minute)
        return fn

    def get_sb_bag_name(self, bag_fullpath):
        basename = self._get_sb_bag_basename(bag_fullpath)
        seq = self.bag_seqs.get(basename, 0) + 1
        self.bag_seqs[basename] = seq
        return basename + "_{}.bag".format(seq)

    def generate_lftp_script(self):
        if not os.path.isdir(self.rosbag_dir):
            return
        self.bag_seqs = {}

        sftp_cmds = [("open -p " + self.port +
                      " -u " + self.account + "," + self.password +
                      " sftp://" + self.sftp_ip)]

        for bag in self.get_bag_files():
            yymmdd = get_bag_yymmdd(bag)
            sftp_dir_name = "{}_{}_camera_{}_1".format(self.company_name, self.vid, yymmdd)
            sb_fn = self.get_sb_bag_name(bag)
            sftp_cmds += ["mkdir -p {}/{}".format(self.vid, sftp_dir_name),
                          "cd {}/{}".format(self.vid, sftp_dir_name),
                          "put -c {} -o {}".format(bag, sb_fn)]
        sftp_cmds += ["bye", ""]
        return "\n".join(sftp_cmds)

    def write_lftp_script(self):
        script = self.generate_lftp_script()
        script_file = os.path.join(self.rosbag_dir, "lftp_script.txt")
        with io.open(script_file, "w") as _fp:
            _fp.write(script_file)
        return script_file

    def run(self):
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            script_file = self.write_lftp_script()
            rospy.logwarn("Write %s", script_file)
            rospy.logwarn("  Use lftp -f %s to upload all the bag files to south bridge server", script_file)
            rate.sleep()
