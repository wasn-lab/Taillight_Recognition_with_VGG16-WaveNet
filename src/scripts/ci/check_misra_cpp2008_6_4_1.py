#!/usr/bin/env python
"""
Check if a given file are comform to coding standard MISRA c++-2008 Rule 6.4.1
Rule 6–4–1 (Required)
    An if ( condition ) construct shall be followed by
    a compound statement. The else keyword shall be
    followed by either a compound statement, or another if
    statement.
"""
import sys
import logging
import os
import subprocess
from ci_utils import (get_affected_files, get_compile_args,
                      is_external_package)


def check_misra_cpp2008_6_4_1_by_cpp(cpp):
    """
    Use clang-tidy to check Rule 6.4.1.
    """
    logging.info("Check MISRA C++-2008 Rule 6.4.1 for %s", cpp)
    args = get_compile_args(cpp)
    if not args:
        return 0
    cmd = ["clang-tidy", cpp,
           "-checks=readability-braces-around-statements",
           "--"] + args
    logging.info(" ".join(cmd))
    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        logging.error("CalledProcessError: %s", " ".join(cmd))

    logging.warning(output.decode("utf-8"))
    return 0


def check_misra_cpp2008_6_4_1():
    """
    Return the number of naming violations
    """
    num_violations = 0
    for cpp in get_affected_files():
        if not os.path.isfile(cpp):
            continue
        if not cpp.endswith(".cpp"):
            continue
        if is_external_package(cpp):
            continue
        num_violations += check_misra_cpp2008_6_4_1_by_cpp(cpp)
    logging.info("Number of violations: %d", num_violations)
    return 0


def main():
    """Prog entry"""
    return check_misra_cpp2008_6_4_1()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
