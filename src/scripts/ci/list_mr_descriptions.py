#!/usr/bin/env python3
"""
List merge request descriptions
"""
import argparse
import requests

ITRIADV_PROJECT_ID = 112
PRIVATE_TOKEN = "LEbxYzsSzycuXkfKto4t"
BASE_URL = "https://gitlab.itriadv.co/api/v4/projects/{}/merge_requests".format(ITRIADV_PROJECT_ID)

def _list_mr_description(sid, eid):
    headers = {"PRIVATE-TOKEN": "LEbxYzsSzycuXkfKto4t"}
    for _id in range(sid, eid+1):
        url = BASE_URL + "/{}".format(_id)
        req = requests.get(url=url, headers=headers)
        data = req.json()
        print("-" * 70)
        print("{}, {} -> {}".format(req.url, data["source_branch"], data["target_branch"]))
        print("")
        print(data["description"])


def main():
    """Prog Entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", default=0, type=int, help="Starting merge request id")
    parser.add_argument("--eid", default=1, type=int, help="Ending merge request id")
    args = parser.parse_args()
    _list_mr_description(args.sid, args.eid)



if __name__ == "__main__":
    main()
