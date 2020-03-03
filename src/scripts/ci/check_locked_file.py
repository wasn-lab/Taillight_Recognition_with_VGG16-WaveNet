#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import sys
import logging
from ci_utils import get_affected_files
__LOCKED_FILES = ["src/CMakeLists.txt"]


def _check_locked_file(affected_files):
    ret = False
    for fname in affected_files:
        if fname in __LOCKED_FILES:
            logging.error("Do not change %s. Contact system admin if you really want to", fname)
            ret = True
    return ret

def main():
    """Prog entry"""
    return 1 if _check_locked_file(get_affected_files()) else 0

if __name__ == "__main__":
    sys.exit(main())
