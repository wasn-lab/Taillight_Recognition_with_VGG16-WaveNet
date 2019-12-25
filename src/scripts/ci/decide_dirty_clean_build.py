#!/usr/bin/env python
"""
Check if a merge request contains package.xml.
"""
from __future__ import print_function
import subprocess


def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


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
    affected_files = _get_affected_files()
    files_trigger_clean_build = _check_package_xml(affected_files)
#    files_trigger_clean_build += _check_cmake(affected_files)
    if files_trigger_clean_build:
        print("Clean build: change in {}".format(" ".join(files_trigger_clean_build)))
    else:
        print("Dirty build: for no change in package.xml.")

if __name__ == "__main__":
    main()
