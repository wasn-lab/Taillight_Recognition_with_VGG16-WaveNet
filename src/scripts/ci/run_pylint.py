#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import subprocess
import sys
import logging

def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


def _run_pylint(affected_files):
    num_fail = 0
    rc_file = "src/scripts/ci/pylintrc"
    for fname in affected_files:
        if not fname.endswith(".py"):
            continue
        cmd = ["pylint", "-E", "--rcfile=" + rc_file, fname]
        print(" ".join(cmd))
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            output = err.output
            num_fail += 1
            logging.error("pylint failed for %s", fname)
        print(output.decode("utf-8"))
    return num_fail

def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    return 1 if _run_pylint(affected_files) > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
