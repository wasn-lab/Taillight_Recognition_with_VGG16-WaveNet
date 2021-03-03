#!/usr/bin/env python3
"""
Calculate bandwith usage for all topics
"""
import argparse
import subprocess
import pexpect
# import rospy


def find_bw(topic):
    cmd = ["rostopic", "bw", topic]
    # print(" ".join(cmd))
    child = pexpect.spawnu(" ".join(cmd))
    try:
        child.expect("window: ", timeout=5)
        output = child.before + child.after
    except pexpect.exceptions.TIMEOUT:
        print("Timeout at topic {}".format(topic))
        output = ""
    child.sendcontrol('c')  # kill child
    child.wait()
    bw = "NA"
    for line in output.splitlines()[::-1]:
        if "average: " in line:
            bw = line
            break
    print("{} {}".format(topic, bw))


def calc_all_bw():
    # get topics by rospy, not work well with python3
    # for topic, _type in rospy.get_published_topics():
    cmd = ["rostopic", "list"]
    output = subprocess.check_output(cmd).decode("utf-8")
    for topic in output.splitlines():
        topic = topic.strip()
        if not topic:
            continue
        find_bw(topic)


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=None)
    args = parser.parse_args()
    topic = args.topic
    if topic:
        find_bw(args.topic)
    else:
        calc_all_bw()

if __name__ == "__main__":
    main()
