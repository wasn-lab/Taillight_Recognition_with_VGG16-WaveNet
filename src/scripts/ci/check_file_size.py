#!/usr/bin/env python
"""
Check if a merge request contains any large file.
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


def _check_fsize(affected_files):
    ret = False
    fsize_limit = 15 * (2**20)
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        fsize = os.path.getsize(fname)
        print("{}: {} bytes".format(fname, fsize))
        if fsize > fsize_limit:
            logging.error("%s: file size too large", fname)
            ret = True
    return ret

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    hit_limit = _check_fsize(affected_files)
    if hit_limit:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
