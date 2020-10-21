import time
import os
import rospkg
import subprocess
import io
import re
import psutil

__BAG_NAME_RGX = re.compile(
    r".*(?P<year>[\d]{4})\-(?P<month>[\d]{2})\-(?P<day>[\d]{2})\-.*\.bag")
_BACKUP_ROSBAG_LFTP_SCRIPT = "/tmp/backup_rosbag_lftp_script.txt"

def _get_stamp_filename(fullpath):
    """.stamp file indicate if we already send the corresponding bag."""
    return fullpath[:-4] + ".stamp"


def _get_backend_password():
    """Use environment variable to avoid leak of password shown in the git repo"""
    passwd = os.environ.get("LFTP_PASSWORD", "")
    if not passwd:
        print(("Please set LFTP_PASSWORD environment variable so that "
               "we can upload rosbags to backend server"))
    return passwd


def _get_bag_ymd(bag):
    """Return the 8-char {year}{month}{day} string encoded in |bag|."""
    match = __BAG_NAME_RGX.match(bag)
    if not match:
        return "20200101"
    year = match.expand(r"\g<year>")
    month = match.expand(r"\g<month>")
    day = match.expand(r"\g<day>")
    return year + month + day



# Send backup rosbag (where abnormal events/states happens) to the backend server
class RosbagSender(object):
    def __init__(self, fqdn, port, user_name, rosbag_backup_dir, vid="itriadv", upload_rate=1000000):
        """
        Currently we use FTP protocol to send rosbags
        """
        self.fqdn = fqdn
        self.port = port
        self.vid = vid
        self.proc = None
        self.user_name = user_name
        self.upload_rate = upload_rate
        if not rosbag_backup_dir:
            self.rosbag_backup_dir = os.path.join(
                os.environ["HOME"], "rosbag_files", "backup")
        else:
            self.rosbag_backup_dir = rosbag_backup_dir

    def run(self):
        while True:
            self.send_if_having_bags()
            time.sleep(15)

    def send_if_having_bags(self):
        if self.proc and self.proc.poll() is None:
            # bag is still uploading
            return
        bags = self.get_unsent_rosbag_filenames()
        if not bags:
            print("No bags to upload")
            return
        print("Send to backend: {}".format(" ".join(bags)))
        self._generate_lftp_scripts(bags)
        self._send_bags()

    def _generate_lftp_scripts(self, bags):
        passwd = _get_backend_password()
        if not passwd:
            return

        ftp_cmds = [
            "set ssl:verify-certificate no",
            "set net:limit-total-rate 0:{}".format(self.upload_rate),
            "open --env-password -p {} -u {} {}".format(self.port, self.user_name, self.fqdn),
        ]
        for bag in bags:
            ymd = _get_bag_ymd(bag)  # backup dir name in backend
            dir_name = "{}/{}".format(self.vid, ymd)
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

    def _send_bags(self):
        shell_cmd = ["lftp", "-f", _BACKUP_ROSBAG_LFTP_SCRIPT]
        self.proc = subprocess.Popen(shell_cmd)

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
