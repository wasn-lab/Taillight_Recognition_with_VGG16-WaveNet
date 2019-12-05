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

def get_mr_list_id_range(headers,_sid, _eid, target_branch="master", state="merged"):
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

    #
    url = BASE_URL + "?"
    if not target_branch is None:
        url += "target_branch=%s" % target_branch
        url += "&"
    if not state is None:
        url += "state=%s" % state
        url += "&"
    #
    for _id in range(_sid, _eid+1):
        url += "iids[]=%d" % _id
        if _id != _eid:
                url += "&"
    req = requests.get(url=url, headers=headers)
    data_list = req.json() # <-- a dict()
    # See if it's end of merge list
    return ( data_list, req )

def get_mr_list_date_range(headers, _s_date, _e_date, target_branch="master", state="merged"):
    """
    Request a list of merge requests in given id range
    input:
        - _sid: start mr-id
        - _eid: end mr-id
        - target_branch
        - state: opened, closed, locked, or merged
        - headers
    output:
        - data_list
        - requested URL
    """
    # url = BASE_URL + "?created_after=2019-11-26T14:44:29.238+08:00"
    # url = BASE_URL + "?created_after=2019-12-5T02:10:00+08:00" # Note: the "+08:00 will be ignored"
    # url = BASE_URL + "?created_after=2019-12-5T02:10:00" #
    # url = BASE_URL + "?created_after=2019-12-4T00:00:00"
    # url = BASE_URL + "?created_after=2019-12-4"

    dT = datetime.timedelta(hours=8) # The time zone of TW
    _s_date_utc = _s_date - dT
    _e_date_utc = _e_date - dT
    #
    url = BASE_URL + "?"
    if not target_branch is None:
        url += "target_branch=%s" % target_branch
        url += "&"
    if not state is None:
        url += "state=%s" % state
        url += "&"
    #
    url += "created_after=%s" % _s_date_utc.strftime("%Y-%m-%dT%H:%M:%S")
    url += "&"
    url += "created_before=%s" % _e_date_utc.strftime("%Y-%m-%dT%H:%M:%S")
    req = requests.get(url=url, headers=headers)
    data_list = req.json() # <-- a dict()

    if len(data_list) == 20:
        _e_date_tail = convert_gitlab_time_to_datetime(data_list[-1]["created_at"])
        _e_date_tail += datetime.timedelta(seconds=1) # Plus 1 second to prevent from missing log due to truncation of second
        print("Narrowing searching range: %s ~ %s" % (_s_date.strftime("%Y-%m-%dT%H:%M:%S"), _e_date_tail.strftime("%Y-%m-%dT%H:%M:%S")))
        data_list_tail, req_tail = get_mr_list_date_range(headers, _s_date, _e_date_tail, target_branch, state)
        id_head = 0
        for _idx in range(len(data_list_tail)):
            _a = data_list[-1]["reference"]
            _b = data_list_tail[_idx]["reference"]
            if _b < _a:
                id_head = _idx
                break
        print("id_head = %d" % id_head)
        data_list += data_list_tail[id_head:]
    # See if it's end of merge list
    return ( data_list, req )

def _list_mr_description(s_date, e_date, target_branch="master"):
    global gitlab_headers

    data_list, req = get_mr_list_date_range(gitlab_headers, s_date, e_date, target_branch)
    print("type(data) = %s" % str(type(data_list)))
    print("len(data) = %d" % len(data_list))
    print("req = %s" % req)
    for data in data_list:
        print("-" * 70)
        print("Merge id: %s" % str(data["reference"]))

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
        # print("\nDescriptions:\n%s" % str(data["description"]))


def string_to_datetime(date_str):
    """
    This is an utility for converting the string to datetime.
    """
    _date = None
    # 2019-12-05T10:30:25.123
    try:
        _date = datetime.datetime.strptime(date_str[:date_str.rfind(".")], "%Y-%m-%dT%H:%M:%S")
        return _date
    except:
        pass
    # 2019-12-05T10:30
    try:
        _date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
        return _date
    except:
        pass
    # 2019-12-05T10
    try:
        _date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H")
        return _date
    except:
        pass
    # 2019-12-05
    try:
        _date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return _date
    except:
        pass
    # 2019-12
    try:
        _date = datetime.datetime.strptime(date_str, "%Y-%m")
        return _date
    except:
        pass
    # 2019
    try:
        _date = datetime.datetime.strptime(date_str, "%Y")
        return _date
    except:
        pass
    return _date

def main():
    """Prog Entry"""
    global gitlab_headers
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", default="2019-12-4T10:00", help="Starting merge request datetime")
    parser.add_argument("-e", "--end", default="2019-12-5T17:00", help="Ending merge request datetime")
    args = parser.parse_args()

    # _s_date = datetime.datetime(2019,12,4,10,00)
    # _e_date = datetime.datetime(2019,12,6)
    # _s_date = datetime.datetime.strptime(args.start, "%Y-%m-%dT%H:%M")
    # _e_date = datetime.datetime.strptime(args.end, "%Y-%m-%dT%H:%M")
    _s_date = string_to_datetime(args.start)
    _e_date = string_to_datetime(args.end)
    print("Start at [%s]" % _s_date.strftime("%Y-%m-%dT%H:%M:%S"))
    print("End at [%s]" % _e_date.strftime("%Y-%m-%dT%H:%M:%S"))

    _list_mr_description(_s_date, _e_date)


    #
    # data_list, req = get_mr_list_date_range(gitlab_headers, _s_date, _e_date)
    # print("type(data) = %s" % str(type(data_list)))
    # print("len(data) = %d" % len(data_list))
    # # print("data = %s" % str(json.dumps(data_list, indent=4)))
    # print("req.url = %s" % req.url)
    #
    # for data in data_list:
    #     print("-" * 70)
    #     print("Merge id: !%s" % str(data["reference"]))
    #     print("created_at: %s" % str(data["created_at"]))
    #     print("merged_at: %s" % str(data["merged_at"]))
    #     print("source_branch: %s" % str(data["source_branch"]) )
    #     print("target_branch: %s" % str(data["target_branch"]) )
    #     # print("data = %s" % str(json.dumps(data, indent=4)))
    #     # print("\nDescriptions:\n%s" % str(data["description"]))





if __name__ == "__main__":
    main()
