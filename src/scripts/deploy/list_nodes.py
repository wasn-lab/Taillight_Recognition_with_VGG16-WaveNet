#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import sys
import os
import json
try: #py3k
    import urllib.parse as urlparse
except ImportError:
    import urlparse
from rosnode import get_node_names
import rosgraph


def __find_machines(nodes):
    master = rosgraph.Master("/rosnode")
    for node in nodes:
        uri = master.lookupNode(node["name"])
        node["machine"] = urlparse.urlparse(uri).hostname
    return nodes


def __find_pub_subs(nodes):
    master = rosgraph.Master("/rosnode")
    state = master.getSystemState()
    for node in nodes:
        node_name = node["name"]
        node["publications"] = [{"name": t} for t, l in state[0] if node_name in l]
        node["subscriptions"] = [{"name": t} for t, l in state[1] if node_name in l]
    return nodes


def __calc_topic_subcriptions(nodes):
    cnt = {}
    for node in nodes:
        for topic in node["publications"]:
            cnt[topic["name"]] = 0
        for topic in node["subscriptions"]:
            cnt[topic["name"]] = 0

    for node in nodes:
        for topic in node["subscriptions"]:
            cnt[topic["name"]] += 1

    for node in nodes:
        for topic in node["publications"]:
            tname = topic["name"]
            topic["num_subscribers"] = cnt[tname]
    return nodes


def list_nodes():
    nodes = [{"name": _} for _ in get_node_names()]

    __find_machines(nodes)
    __find_pub_subs(nodes)
    __calc_topic_subcriptions(nodes)
    print(json.dumps(nodes, indent=2, sort_keys=True))


def main():
    master = os.environ.get("ROS_MASTER_URI", "")
    print("ROS_MASTER_URI: " + master)
    list_nodes()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
