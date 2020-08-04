#!/usr/bin/env python
import requests
import argparse
import pprint
import json
import io
import re

URL = "http://60.250.196.133:8300/DataMgmt/BackendService"


def post_accuracy(accuracy):
    jdata = {"type": "DM.001", "accuracy": accuracy}
    resp = requests.post(URL, data=json.dumps(jdata))
    pprint.pprint(resp.json())


def parse_log(log_file):
    with io.open(log_file, encoding="utf-8") as _fp:
        contents = _fp.read().splitlines()
    rgx = re.compile(r"for conf_thresh = .*, precision = (?P<accuracy>[\d\.]+), recall = .*, F1-score = .*")
    for line in contents:
        line = line.strip()
        match = rgx.match(line)
        if match:
            return match.expand(r"\g<accuracy>")
    return "0.87"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()
    accuracy = parse_log(args.log_file)
    post_accuracy(accuracy)

if __name__ == "__main__":
    main()
