#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import subprocess
import os
import sys
import logging
LOCKED_FILES = ["src/CMakeLists.txt"]

def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


def _check_locked_file(affected_files):
    ret = False
    for fname in affected_files:
        if fname in LOCKED_FILES:
            logging.error("Do not change %s. Contact system admin if you really want to", fname)
            ret = True
    return ret

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    return 1 if _check_locked_file(affected_files) else 0

if __name__ == "__main__":
    sys.exit(main())
