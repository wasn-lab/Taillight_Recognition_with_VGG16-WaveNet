#!/usr/bin/env python
import argparse
import configparser
import io
import os
import re
import logging
import subprocess
import time

BAG_RGX = re.compile(
    r".+_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})_[\d]+.bag")


def _get_ini_filename():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ini_file = os.path.join(cur_dir, "sb.ini")
    if not os.path.isfile(ini_file):
        logging.error("Cannot find ini file: %s", ini_file)
    return ini_file


def get_sb_config():
    cfg = configparser.ConfigParser()
    cfg.read(_get_ini_filename())
    return {key: cfg["south_bridge"][key] for key in cfg["south_bridge"]}


def get_default_bag_dir():
    return os.path.join(os.environ["HOME"], "rosbag_files", "tmp")


def get_bag_filenames(bag_dir):
    ret = []
    for root, _dirs, files in os.walk(bag_dir):
        for fname in files:
            if not fname.endswith(".bag"):
                continue
            ret.append(os.path.join(root, fname))
    ret.sort()
    return ret


def convert_to_sb_bag_name(bag_fullpath, seq):
    path, bag_fn = os.path.split(bag_fullpath)
    match = BAG_RGX.search(bag_fn)
    if not match:
        logging.warn("Cannot parse filename: %s", bag_fullpath)
        return bag_fullpath
    year = match.expand(r"\g<year>")
    month = match.expand(r"\g<month>")
    day = match.expand(r"\g<day>")
    hour = match.expand(r"\g<hour>")
    minute = match.expand(r"\g<minute>")
    second = match.expand(r"\g<second>")
    cfg = get_sb_config()
    fn = (cfg["company_name"] + "_" + cfg["vid"] + "_camera_" +
          year + month + day + "_" + hour + minute +
          "_{}".format(seq) + ".bag")
    return fn


def get_bag_yymmdd(bag):
    path, bag_fn = os.path.split(bag)
    match = BAG_RGX.search(bag_fn)
    if not match:
        logging.warn("Cannot parse filename: %s", bag_fullpath)
        return "999999"
    year = match.expand(r"\g<year>")
    month = match.expand(r"\g<month>")
    day = match.expand(r"\g<day>")
    return year + month + day


def gen_sftp_cmd(bag, idx):
    """
    Input:
    bag -- full path of a rosbag
    idx -- Index (used for generating serial number)
    """
    yymmdd = get_bag_yymmdd(bag)
    cfg = get_sb_config()
    sftp_dir_name = "{}_{}_camera_{}_1".format(cfg["company_name"], cfg["vid"], yymmdd)
    dest_bag = convert_to_sb_bag_name(bag, idx)
    sftp_cmds = [
        "open -p " + cfg["port"] + " -u " + cfg["account"] + "," + cfg["password"] + " sftp://{}".format(cfg["sftp_ip"]),
        "mkdir -p {}/{}".format(cfg["vid"], sftp_dir_name),
        "cd {}/{}".format(cfg["vid"], sftp_dir_name),
        "put -c {} -o {}".format(bag, dest_bag),
        "bye",
        ""]
    return "\n".join(sftp_cmds)


def upload_with_sftp(bag, idx):
    """
    Input:
    bag -- full path of a rosbag
    idx -- Index (used for generating serial number)
    """
    lftp_script = "/tmp/{}.txt".format(time.time())
    with io.open(lftp_script, "w") as _fp:
        _fp.write(gen_sftp_cmd(bag, idx))

    try:
        subprocess.call(["lftp", "-f", lftp_script])
        os.unlink(lftp_script)
        logging.warn("Done uploading %s", bag)
    except subprocess.CalledProcessError:
        logging.warn("Upload failed, see %s for lftp commands", lftp_script)


def upload_bags(rosbag_dir):
    bags = get_bag_filenames(rosbag_dir)
    for idx, bag in enumerate(bags):
        upload_with_sftp(bag, idx)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rosbag-dir", default=get_default_bag_dir())
    args = parser.parse_args()
    upload_bags(args.rosbag_dir)


if __name__ == "__main__":
    main()
