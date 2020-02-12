#!/usr/bin/env python
"""
Utility functions for CI scripts
"""
from functools import lru_cache
import io
import subprocess


@lru_cache(maxsize=0)
def get_affected_files():
    """ Return a list of files in the active merge request. """
    cmd = ["git", "merge-base", "origin/master", "HEAD"]
    ref_commit = subprocess.check_output(cmd).strip()
    cmd = ["git", "diff", "--name-only", ref_commit]
    output = subprocess.check_output(cmd).decode("utf-8")
    return [fname.strip() for fname in output.splitlines()]


@lru_cache(maxsize=0)
def get_external_pacakges():
    """ Return a list of directories that are external packages. """
    with io.open("src/scripts/ci/external_packages.txt") as _fp:
        contents = _fp.read().splitlines()
    return [_.strip() for _ in contents]


def is_external_package(fname):
    """ Return True if |fname| belongs to external packages. False otherwise"""
    for path in get_external_pacakges():
        if fname.startswith(path):
            return True
    return False
