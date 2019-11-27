#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import subprocess
import os
import sys
import logging
import io


def _get_rw_exts():
    with io.open("src/scripts/ci/rw_files.txt") as _fp:
        contents = _fp.read().splitlines()
    return set([_.strip() for _ in contents])


def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


def _rw_file_marking_executable(fname, rw_exts):
    should_be_rw = False
    for ext in rw_exts:
        if fname.endswith(ext):
            should_be_rw = True
    if not should_be_rw:
        return False
    else:
        return os.access(fname, os.X_OK)

def _check_fmod(affected_files):
    rw_exts = _get_rw_exts()
    exes = []
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if _rw_file_marking_executable(fname, rw_exts):
            exes.append(fname)
    if exes:
        logging.error("File type is executable: %s", " ".join(exes))
        print("Run")
        print("  chmod 644 {}".format(" ".join(exes)))
    else:
        print("file mode check passed.")
    return len(exes)

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    return _check_fmod(affected_files)

if __name__ == "__main__":
    sys.exit(main())
