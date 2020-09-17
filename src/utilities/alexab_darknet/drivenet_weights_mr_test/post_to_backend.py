#!/usr/bin/env python
import argparse
import pprint
import io
import requests

URL = "http://60.250.196.133:8300/DataMgmt/BackendService"


def _post_to_backend(json_result):
    with io.open(json_result, encoding="utf-8") as _fp:
        contents = _fp.read()
    resp = requests.post(URL, data=contents)
    pprint.pprint(resp.json())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-result", required=True)
    args = parser.parse_args()
    _post_to_backend(args.json_result)


if __name__ == "__main__":
    main()
