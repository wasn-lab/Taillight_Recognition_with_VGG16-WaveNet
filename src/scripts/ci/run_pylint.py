#!/usr/bin/env python
"""
Check if a merge request contains any large file.
"""
from __future__ import print_function
import os
import subprocess
import sys
import logging
from ci_utils import get_affected_files, is_external_package, get_repo_path


def _run_pylint(affected_files):
    num_fail = 0
    repo_path = get_repo_path()
    rc_file = os.path.join(repo_path, "src/scripts/ci/pylintrc")
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if not fname.endswith(".py"):
            continue
        if is_external_package(fname):
            continue
        path, py_src = os.path.split(fname)
        os.chdir(path)
        cmd = ["pylint", "-E", "--rcfile=" + rc_file, "--disable=import-error", py_src]
        print(" ".join(cmd))
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            output = err.output
            num_fail += 1
            logging.error("pylint failed for %s", fname)
        output = output.decode("utf-8")
        if output:
            print(output)
        os.chdir(repo_path)
    return num_fail

def main():
    """Prog entry"""
    affected_files = get_affected_files()
    return 1 if _run_pylint(affected_files) > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
