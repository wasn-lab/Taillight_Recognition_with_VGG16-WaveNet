#!/usr/bin/env python
"""
List all package names in the repository.
"""
from __future__ import print_function
import logging
import os
import io
import re
REPO_SRC = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))

def _get_package_dirs():
    ret = []
    for root, _, files in os.walk(REPO_SRC):
        for fname in files:
            if fname != "package.xml":
                continue
            ret.append(root)
    return ret


def _check_cmakelists(pkg_dir):
    cmakelist = os.path.join(pkg_dir, "CMakeLists.txt")
    if not os.path.isfile(cmakelist):
        logging.warning("No CMakeLists.txt in %s", pkg_dir)
        return 0
    with io.open(cmakelist, encoding="utf-8") as _fp:
        contents = _fp.read()
    valid = False
    for line in contents.splitlines():
        if "include(CompilerFlags)" in line:
            valid = True
    if not valid:
        logging.error("%s: does not include CompilerFlags.cmake", pkg_dir)
        return 1
    return 0


def main():
    """Prog entry."""
    for _dir in _get_package_dirs():
        _check_cmakelists(_dir)

if __name__ == "__main__":
    main()
