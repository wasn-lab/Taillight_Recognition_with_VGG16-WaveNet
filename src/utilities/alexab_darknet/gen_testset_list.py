#!/usr/bin/env python
import sys
import os
import io
import argparse
import logging
import hashlib

def md5(fname):
    """Compute md5 hash of the specified file"""
    hasher = hashlib.md5()
    with io.open(fname, "rb") as _fp:
        contents = _fp.read()
    hasher.update(contents)
    return hasher.hexdigest()


def __generate_image_list(testset_dir):
    ret = {}
    for root, _dirs, files in os.walk(testset_dir):
        for fname in files:
            if "jpg" not in fname:
                continue

            fullpath = os.path.abspath(os.path.join(root, fname))
            if "test_dataset" not in fullpath.lower():
                continue
            txt = fullpath[:-3] + "txt"
            if not os.path.isfile(txt):
                logging.error("No such file: %s", txt)
                continue
            ret[md5(fullpath)] = fullpath
    return list(ret.values())


def __generate_list(testset_dir):
    if not os.path.isdir(testset_dir):
        logging.error("No such directory: %s", testset_dir)
        sys.exit(1)
    image_list = __generate_image_list(testset_dir)
    output_file = os.path.join(testset_dir, "valid.txt")
    with io.open(output_file, "w") as _fp:
        _fp.write("\n".join(image_list))
        _fp.write("\n")
    logging.warning("Write %d lines in %s", len(image_list), output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset-dir", required=True)
    args = parser.parse_args()
    __generate_list(args.testset_dir)

if __name__ == "__main__":
    main()
