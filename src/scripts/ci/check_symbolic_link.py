#!/usr/bin/env python
"""
Check if commit files contain symbolic links.
"""
import sys
import logging
import subprocess
import os


def _get_affected_files():
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


def _check_symbolic_links(affected_files):
    num_links = 0
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if os.path.islink(fname):
            logging.error("Do not commit symbolic links: %s", fname)
            num_links += 1
    return num_links


def main():
    """Prog entry"""
    affected_files = _get_affected_files()
    return _check_symbolic_links(affected_files)


if __name__ == "__main__":
    sys.exit(main())
