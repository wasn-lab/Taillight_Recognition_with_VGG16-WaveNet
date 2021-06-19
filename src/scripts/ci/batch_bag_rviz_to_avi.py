#!/usr/bin/env python
import os
import argparse
import logging
import shutil
import subprocess
from bag_jpgs_to_avi import _save_avi

_TMP_DIR = "/dev/shm"

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
        logging.warn("%s exists.", avi_full_path)
        return
    logging.warn("cp %s %s", bag_full_path, _TMP_DIR)
    shutil.copy(bag_full_path, _TMP_DIR)
    if bag_full_path.endswith(".gz"):
        _path, filename = os.path.split(bag_full_path)
        cmd = ["gzip", "-d", os.path.join(_TMP_DIR, filename)]
        cmd_str = " ".join(cmd)
        try:
            logging.warn(cmd_str)
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            logging.warn("Cannot run %s", cmd_str)
    temp_bag_full_path = _get_tmp_bag_full_path(bag_full_path)
    _save_avi(temp_bag_full_path, "/xwin_grabber/rviz/jpg", avi_full_path)
    if os.path.isfile(temp_bag_full_path):
        logging.warn("rm %s", temp_bag_full_path)
        os.unlink(temp_bag_full_path)


def _find_bags(dirname):
    ret = []
    for root, _dirs, _files in os.walk(dirname):
        for filename in _files:
            if filename.endswith(".bag") or filename.endswith(".bag.gz"):
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
