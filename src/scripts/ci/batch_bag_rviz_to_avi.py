#!/usr/bin/env python
import os
import argparse
import logging
import shutil
import subprocess
import datetime
from bag_jpgs_to_avi import _save_avi

_TMP_DIR = "/tmp"

def _get_avi_full_path(bag_full_path):
    path, filename = os.path.split(bag_full_path)
    basename = filename.split(".")[0]
    return os.path.join(path, basename + ".avi")


def _get_tmp_bag_full_path(bag_full_path):
    _path, filename = os.path.split(bag_full_path)
    basename = filename.split(".")[0]
    return os.path.join(_TMP_DIR, basename + ".bag")


def _rviz2avi(bag_full_path):
    avi_full_path = _get_avi_full_path(bag_full_path)
    if os.path.isfile(avi_full_path):
        logging.warning("%s exists.", avi_full_path)
        return
    logging.warning("cp %s %s", bag_full_path, _TMP_DIR)
    shutil.copy(bag_full_path, _TMP_DIR)
    if bag_full_path.endswith(".gz"):
        _path, filename = os.path.split(bag_full_path)
        cmd = ["gzip", "-d", os.path.join(_TMP_DIR, filename)]
        cmd_str = " ".join(cmd)
        try:
            logging.warning(cmd_str)
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            logging.warning("Cannot run %s", cmd_str)
    temp_bag_full_path = _get_tmp_bag_full_path(bag_full_path)
    _save_avi(temp_bag_full_path, "/xwin_grabber/rviz/jpg", avi_full_path)
    if os.path.isfile(temp_bag_full_path):
        logging.warning("rm %s", temp_bag_full_path)
        os.unlink(temp_bag_full_path)


def _is_recent_bag(bag_fullpath):
    path, filename = os.path.split(bag_fullpath)
    # consider the case auto_record_2021-06-21-23-41-09_66.bag
    fields = filename.split("_")
    if len(fields) < 3:
        logging.warning("Invalid bag filename: %s", filename)
        return False
    dt_fields = fields[2].split("-")
    if len(dt_fields) != 6:
        logging.warning("Cannot get datetime fields: %s", filename)
        return False
    year = int(dt_fields[0])
    month = int(dt_fields[1])
    day = int(dt_fields[2])

    dt_bag = datetime.datetime(year, month, day)
    delta = datetime.datetime.now() - dt_bag
    return bool(delta.days <= 7)


def _find_bags(dirname):
    ret = []
    for root, _dirs, _files in os.walk(dirname):
        for filename in _files:
            if filename.endswith(".bag") or filename.endswith(".bag.gz"):
                if _is_recent_bag(filename):
                    ret.append(os.path.join(root, filename))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-name", "-d", required=True)
    args = parser.parse_args()
    bags = _find_bags(args.dir_name)
    for bag_full_path in bags:
        _rviz2avi(bag_full_path)

if __name__ == "__main__":
    main()
