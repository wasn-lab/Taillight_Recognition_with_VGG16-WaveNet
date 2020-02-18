#!/usr/bin/env python
"""
Check if a given file are comform to coding standard MISRA c++-2008 Rule 6.4.1
Rule 6–4–1 (Required)
    An if ( condition ) construct shall be followed by
    a compound statement. The else keyword shall be
    followed by either a compound statement, or another if
    statement.
"""
import argparse
import sys
import logging
import os
import subprocess
from ci_utils import get_compile_args, is_external_package


def check_misra_cpp2008_6_4_1_by_cpp(cpp, apply_fix):
    """
    Use clang-tidy to check Rule 6.4.1.
    """
    logging.info("Check MISRA C++-2008 Rule 6.4.1 for %s", cpp)
    args = get_compile_args(cpp)
    if not args:
        return
    fix = "--fix" if apply_fix else ""
    cmd = ["clang-tidy", cpp, fix,
           "-checks=-*,readability-braces-around-statements",
           "--"] + args
    logging.info(" ".join(cmd))
    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        logging.error("CalledProcessError: %s", " ".join(cmd))

    if not output:
        return
    logging.warning(output.decode("utf-8"))
    logging.info("If Errors arise, run \n"
                 "    python src/scripts/ci/check_misra_cpp2008_6_4_1.py --fix "
                 "--cpp %s\nto fix it", cpp)


def check_misra_cpp2008_6_4_1(cpp, apply_fix):
    """
    Return the number of naming violations
    """
    if not os.path.isfile(cpp):
        return 0
    if not cpp.endswith(".cpp"):
        return 0
    if is_external_package(cpp):
        return 0
    check_misra_cpp2008_6_4_1_by_cpp(cpp, apply_fix)
    return 0


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True)
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    return check_misra_cpp2008_6_4_1(args.cpp, args.fix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
