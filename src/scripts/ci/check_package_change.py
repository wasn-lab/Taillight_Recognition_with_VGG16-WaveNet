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
    for fname in affected_files:
        if "package.xml" in fname:
            print("package.xml modified")

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    _check_package_xml(affected_files)

if __name__ == "__main__":
    main()
