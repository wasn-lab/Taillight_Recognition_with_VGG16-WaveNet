#!/usr/bin/env python
import argparse
import io
import logging
import json
import os
import sys

def _read_json_file(jfile):
    with io.open(jfile, encoding="utf-8") as _fp:
        contents = _fp.read()
    return json.loads(contents)


def _merge_check_result(artifacts_dir):
    docs = []
    for angle in ["fov60", "fov120"]:
        jfile = os.path.join(artifacts_dir, angle, "check_weights_result.json")
        if not os.path.isfile(jfile):
            continue
        docs += _read_json_file(jfile)
    num_violations = sum(_["num_violations"] for _ in docs)
    result = {"test_cases": docs,
              "num_violations": num_violations,
              "result": "PASS",
              }
    if num_violations > 0:
        result["result"] = "FAIL"
    output_file = os.path.join(artifacts_dir, "check_result.json")
    with io.open(output_file, "w", encoding="utf-8") as _fp:
        _fp.write(json.dumps(result, sort_keys=True))
    logging.warn("Write %s", output_file)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    args = parser.parse_args()
    result = _merge_check_result(args.artifacts_dir)
    if result["num_violations"] > 0:
        logging.error("Find unexpected detection!")
    return result["num_violations"]


if __name__ == "__main__":
    sys.exit(main())
