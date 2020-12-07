#!/usr/bin/env python
import requests
import argparse
import pprint
import json
import io
import re

URL = "http://118.163.54.112:8300/DataMgmt/BackendService"


def post_accuracy(accuracy, _id):
    jdata = {"type": "DM.001", "accuracy": accuracy, "id": _id}
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
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()
    accuracy = parse_log(args.log_file)
    post_accuracy(accuracy, args.id)

if __name__ == "__main__":
    main()
