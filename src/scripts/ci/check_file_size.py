#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import os
import sys
import logging
from ci_utils import get_affected_files


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
    hit_limit = _check_fsize(get_affected_files())
    if hit_limit:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
