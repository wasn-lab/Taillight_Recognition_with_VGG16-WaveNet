#!/usr/bin/env python
"""
Call clang-tidy to help review
"""
import argparse
import sys
import logging
import os
import subprocess
from ci_utils import is_external_package


def __run_clang_tidy(cpp, apply_fix, build_path):
    """
    Use clang-tidy to check Rule 6.4.1.
    """
    fix = "--fix" if apply_fix else ""
    cmd = ["clang-tidy", fix,
           "-p", build_path,
           "-checks=-*,readability-braces-around-statements",
           cpp]
    try:
        output = subprocess.check_output(cmd)
        if output:
            logging.warning(output.decode("utf-8"))
    except subprocess.CalledProcessError as _e:
        logging.error("CalledProcessError: %s\n%s",
                      " ".join(_e.cmd), _e.output.decode("utf-8"))


def check_misra_cpp2008_6_4_1(cpp, apply_fix, build_path):
    """
    Return the number of naming violations
    """
    if not os.path.isfile(cpp):
        return 0
    if not cpp.endswith(".cpp"):
        return 0
    if is_external_package(cpp):
        return 0
    __run_clang_tidy(cpp, apply_fix, build_path)
    return 0


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--build-path", default="build_clang")
    args = parser.parse_args()
    return check_misra_cpp2008_6_4_1(args.cpp, args.fix, args.build_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
