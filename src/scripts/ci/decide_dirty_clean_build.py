#!/usr/bin/env python
"""
Check if a merge request contains package.xml.
"""
from __future__ import print_function
from ci_utils import get_affected_files


def _check_package_xml(affected_files):
    ret = []
    for fname in affected_files:
        if "package.xml" in fname:
            ret.append(fname)
    return ret


def _check_cmake(affected_files):
    ret = []
    for fname in affected_files:
        if ".cmake" in fname or "CMakeLists.txt" in fname:
            ret.append(fname)
    return ret


def main():
    """Prog entry"""
    affected_files = get_affected_files()
    files_trigger_clean_build = _check_package_xml(affected_files)
#    files_trigger_clean_build += _check_cmake(affected_files)
    if files_trigger_clean_build:
        print("Clean build: change in {}".format(" ".join(files_trigger_clean_build)))
    else:
        print("Dirty build: for no change in package.xml.")

if __name__ == "__main__":
    main()
