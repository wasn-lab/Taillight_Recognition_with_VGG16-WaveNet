#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import os
import sys
import logging
import io
from ci_utils import get_affected_files


def _get_rw_exts():
    with io.open("src/scripts/ci/rw_files.txt") as _fp:
        contents = _fp.read().splitlines()
    return {_.strip() for _ in contents}


def _rw_file_marking_executable(fname, rw_exts):
    should_be_rw = False
    for ext in rw_exts:
        if fname.endswith(ext):
            should_be_rw = True
    if not should_be_rw:
        return False
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
    return _check_fmod(get_affected_files())

if __name__ == "__main__":
    sys.exit(main())
