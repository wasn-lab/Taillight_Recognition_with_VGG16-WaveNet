#!/usr/bin/env python
"""
Check if commit files contain symbolic links.
"""
import sys
import logging
import argparse
import os
import io
import re


def _analyze_tf_args(tf_args):
    fields = tf_args.split()
    if len(fields) != 8:
        logging.error("Invalid tf args: %s", tf_args)
        return 1
    for str_num in fields[:6]:
        try:
            _ = float(str_num)
        except ValueError:
            logging.error("%s is a invalid number", str_num)
            return 1
    return 0


def _check_tf_args(fname):
    """
    tf args should be in the following form:
    <node pkg="tf2_ros" type="static_transform_publisher" args="0 0 0 -3.1415926 0 0 /base_link /os1_sensor" />
    """
    num_failure = 0
    with io.open(fname, encoding="utf-8") as _fp:
        contents = _fp.read()
    rgx = re.compile(r"args=\"(?P<args>[^\"]+)\"")
    for line in contents.splitlines():
        if "static_transform_publisher" not in line or "tf2_ros" not in line:
            continue
        match = rgx.search(line)
        if not match:
            continue
        args = match.expand(r"\g<args>")
        num_failure += _analyze_tf_args(args)
    return num_failure


def _check_launch(fname):
    if not os.path.isfile(fname):
        logging.warning("%s: file not found", fname)
        return 0
    if not fname.endswith(".launch"):
        logging.info("%s: This script checks only *.launch", fname)
        return 0
    return _check_tf_args(fname)


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True)
    args = parser.parse_args()
    fname = args.file
    return _check_launch(fname)


if __name__ == "__main__":
    sys.exit(main())
