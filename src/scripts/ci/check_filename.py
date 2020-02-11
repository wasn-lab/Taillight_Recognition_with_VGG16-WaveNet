#!/usr/bin/env python
"""
Check if a merge request contain files whose name contain space.
"""
from __future__ import print_function
import subprocess
import os
import sys
import logging


def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


def _check_space(affected_files):
    fnames_with_space = []
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if " " in fname:
            fnames_with_space.append(fname)
    if fnames_with_space:
        logging.error("File name contains space: %s", " ".join(fnames_with_space))
    return len(fnames_with_space)

def _check_hpp(affected_files):
    fnames_hpp = []
    for fname in affected_files:
        if fname.endswith(".hpp"):
            fnames_hpp.append(fname)
    if fnames_hpp:
        logging.error("ROS header files are named after .h, not.hpp: %s", " ".join(fnames_hpp))
    return len(fnames_hpp)

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    return _check_space(affected_files) + _check_hpp(affected_files)

if __name__ == "__main__":
    sys.exit(main())
