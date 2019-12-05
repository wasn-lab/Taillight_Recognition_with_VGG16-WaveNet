#!/usr/bin/env python3
"""
List merge request descriptions
"""
import argparse
import requests
#
import datetime
import re # Regular expressions
import json


ITRIADV_PROJECT_ID = 112
PRIVATE_TOKEN = "LEbxYzsSzycuXkfKto4t"
BASE_URL = "https://gitlab.itriadv.co/api/v4/projects/{}/merge_requests".format(ITRIADV_PROJECT_ID)
gitlab_headers = {"PRIVATE-TOKEN": "LEbxYzsSzycuXkfKto4t"}

def is_empty_mr(data):
    """
    This is a wrapper for checking if the merge-request is empty (not exist).
    Note:
    - The key "reference" is the id of merge-request
    """
    return not ("reference" in data)

def is_valid_mr(data):
    """
    This is a wrapper for checking if the merge-request is valid (merged).
    Note:
    - The key "merged_at" is the time stamp for merged mr
    """
    return (data["merged_at"] is not None)

def convert_gitlab_time_to_datetime(time_gitlab):
    """
    This is a tool for converting the string of time form gitlab to datatime object
    """
    # Note: remove the things after "."
    return datetime.datetime.strptime(time_gitlab[:time_gitlab.rfind(".")], "%Y-%m-%dT%H:%M:%S")

def get_mr(_id, headers):
    """
    Request a merge request as given id
    input:
        - _id, headers
    output:
        - succeed?
        - data, if succeed
        - requested URL
    """
    url = BASE_URL + "/{}".format(_id)
    req = requests.get(url=url, headers=headers)
    data = req.json() # <-- a dict()
    # See if it's end of merge list
    return ( not is_empty_mr(data), data, req )

def get_mr_list_id_range(_sid, _eid, headers):
    """
    Request a list of merge requests in given id range
    input:
        - _sid: start mr-id
        - _eid: end mr-id
        - headers
    output:
        - data_list
        - requested URL
    """
    # url = BASE_URL + "?created_after=2019-11-26T14:44:29.238+08:00"
    # url = BASE_URL + "?iids[]=100&iids[]=110"
    url = BASE_URL + "?"
    for _id in range(_sid, _eid+1):
        url += "iids[]=%d" % _id
        if _id != _eid:
                url += "&"
    req = requests.get(url=url, headers=headers)
    data_list = req.json() # <-- a dict()
    # See if it's end of merge list
    return ( data_list, req )

def get_mr_list_date_range(_s_date, _e_date, headers):
    """
    Request a list of merge requests in given id range
    input:
        - _sid: start mr-id
        - _eid: end mr-id
        - headers
    output:
        - data_list
        - requested URL
    """
    # url = BASE_URL + "?created_after=2019-11-26T14:44:29.238+08:00"
    # url = BASE_URL + "?iids[]=100&iids[]=110"
    url = BASE_URL + "?"
    for _id in range(_sid, _eid+1):
        url += "iids[]=%d" % _id
        if _id != _eid:
                url += "&"
    req = requests.get(url=url, headers=headers)
    data_list = req.json() # <-- a dict()
    # See if it's end of merge list
    return ( data_list, req )

def _list_mr_description(sid, eid):
    global gitlab_headers

    data_list, req = get_mr_list_id_range(sid, eid, gitlab_headers)
    print("type(data) = %s" % str(type(data_list)))
    print("len(data) = %d" % len(data_list))
    print("req = %s" % req)
    for data in data_list:
        print("-" * 70)
        print("Merge id: !%s" % str(data["reference"]))

        # analyize data
        #-----------------------------#
        if not is_valid_mr(data):
            print("This merged-request is not yet merged.")
            continue
        #
        time_created = convert_gitlab_time_to_datetime(data["created_at"])
        time_merged = convert_gitlab_time_to_datetime(data["merged_at"])
        # time_merged_formate = time_merged.strftime("%Y-%m-%d-%H-%M-%S")
        # print("time_merged_formate = %s" % time_merged_formate)

        print("created_at: %s" % str(data["created_at"]))
        # print("type = %s" % str(type(data["created_at"])))
        print("merged_at: %s" % str(data["merged_at"]))
        print("source_branch: %s" % str(data["source_branch"]) )
        print("target_branch: %s" % str(data["target_branch"]) )
        # print("data = %s" % str(json.dumps(data, indent=4)))
        print("\nDescriptions:\n%s" % str(data["description"]))
        # print(data["description"])




def main():
    """Prog Entry"""
    global gitlab_headers
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", default=0, type=int, help="Starting merge request id")
    parser.add_argument("--eid", default=1, type=int, help="Ending merge request id")
    args = parser.parse_args()
    _list_mr_description(args.sid, args.eid)





if __name__ == "__main__":
    main()
