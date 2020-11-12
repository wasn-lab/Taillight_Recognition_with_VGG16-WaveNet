"""
Send backup rosbag files to backend.
"""
from __future__ import print_function
import time
import os
import io
import subprocess
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
        self.fqdn = fqdn
        self.port = port
        self.license_plate_number = get_license_plate_number()
        self.proc = None
        self.user_name = user_name
        self.password = password
        self.upload_rate = upload_rate
        if not rosbag_backup_dir:
            self.rosbag_backup_dir = os.path.join(
                os.environ["HOME"], "rosbag_files", "backup")
        else:
            self.rosbag_backup_dir = rosbag_backup_dir
        print("rosbag backup dir: {}".format(self.rosbag_backup_dir))
        print("backend server fqdn: {}, user_name: {}".format(self.fqdn, self.user_name))

    def run(self):
        while True:
            self.send_if_having_bags()
            time.sleep(1)

    def send_if_having_bags(self):
        bags = self.get_unsent_rosbag_filenames()
        if not bags:
            print("No bags to upload")
            return
        self.send_bags(bags)

    def send_bags(self, bags):
        bags.sort()
        for bag in bags:
            print("notify backend: {} is ready to be uploaded".format(bag))
            notify_backend_with_new_bag(bag)
            lftp_script_filename = self._generate_lftp_script(bag)
            ret = _send_bag_by_lftp(lftp_script_filename)
            if ret == 0:
                print("notify backend: {} has been uploaded successfuly".format(bag))
                notify_backend_with_uploaded_bag(bag)

    def _generate_lftp_script(self, bag):
        ftp_cmds = [
            "set ssl:verify-certificate no",
            "set net:limit-total-rate 0:{}".format(self.upload_rate),
            "open -p {} -u {},{} {}".format(self.port, self.user_name, self.password, self.fqdn),
        ]
        ymd = get_bag_yymmdd(bag)  # backup dir name in backend
        dir_name = "/{}/{}".format(self.license_plate_number, ymd)
        ftp_cmds += [
            "mkdir -p {}".format(dir_name),
            "cd {}".format(dir_name),
            "put -c {}".format(bag),
            "!touch {}".format(_get_stamp_filename(bag)),
        ]
        ftp_cmds += ["bye"]

        if os.path.isfile(_BACKUP_ROSBAG_LFTP_SCRIPT):
            os.unlink(_BACKUP_ROSBAG_LFTP_SCRIPT)

        with open(_BACKUP_ROSBAG_LFTP_SCRIPT, "w") as _fp:
            _fp.write("\n".join(ftp_cmds))
            _fp.write("\n")
        return _BACKUP_ROSBAG_LFTP_SCRIPT

    def get_unsent_rosbag_filenames(self):
        """Return a list of bag files that are not sent back"""
        files = os.listdir(self.rosbag_backup_dir)
        ret = []
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
