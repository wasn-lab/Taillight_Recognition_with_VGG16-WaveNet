#!/usr/bin/env python
"""
Check if a merge request contain files whose name contain space.
"""
from __future__ import print_function
import os
import sys
import logging
from ci_utils import get_affected_files, is_external_package


def _check_space(affected_files):
    fnames_with_space = []
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if " " in fname:
            fnames_with_space.append(fname)
    if fnames_with_space:
        logging.error("File name contains space: %s",
                      " ".join(fnames_with_space))
    return len(fnames_with_space)


def _check_hpp(affected_files):
    fnames_hpp = []
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if is_external_package(fname):
            continue
        if fname.endswith(".hpp"):
            fnames_hpp.append(fname)
    if fnames_hpp:
        logging.error(
            "ROS header files are named after .h, not.hpp: %s\n"
            "See http://wiki.ros.org/CppStyleGuide, Sec 4.3. "
            "Skip .hpp check for external packages by modifying "
            "src/scripts/ci/external_packages.txt",
            " ".join(fnames_hpp))
    return len(fnames_hpp)


def _check_artifacts(affected_files):
    violations = []
    for fname in affected_files:
        if not os.path.isfile(fname):
            continue
        if "CMakeFiles" not in fname:
            continue
        if fname.endswith(".o") or fname.endswith(".a"):
            violations.append(fname)
    if violations:
        logging.error("The commit contains files inside CMakeFiles: %s",
                      " ".join(violations))
    return len(violations)


def main():
    """Prog entry"""
    affected_files = get_affected_files()
    ret = _check_space(affected_files)
    # ret += _check_hpp(affected_files)
    ret += _check_artifacts(affected_files)
    return ret


if __name__ == "__main__":
    sys.exit(main())
