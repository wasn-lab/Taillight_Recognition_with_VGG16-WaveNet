#!/usr/bin/env python
import requests
import argparse
import pprint
import json

URL = "http://60.250.196.133:8300/DataMgmt/BackendService"


def post_accuracy(accuracy):
    jdata = {"type": "DM.001", "accuracy": accuracy}
    resp = requests.post(URL, data=json.dumps(jdata))
    pprint.pprint(resp.json())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=float, required=True)
    args = parser.parse_args()
    post_accuracy(args.accuracy)

if __name__ == "__main__":
    main()
