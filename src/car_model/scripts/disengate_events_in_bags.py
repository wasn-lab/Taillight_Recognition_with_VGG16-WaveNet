from __future__ import print_function
import argparse
import datetime
import rosbag

TOPIC = "/vehicle/report/itri/fail_safe_status"


def _has_disengage_event(msg):
    if "Disengage: " in msg.data:
        return True
    return False


def _analyze(bag_filename):
    bag = rosbag.Bag(bag_filename)
    for _topic, msg, timestamp in bag.read_messages(topics=[TOPIC]):
        if _has_disengage_event(msg):
            dt_obj = datetime.datetime.fromtimestamp(timestamp.secs)
            print("{}: Disengage event at {}".format(bag_filename, dt_obj))
    bag.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", required=True)
    args = parser.parse_args()
    _analyze(args.rosbag)

if __name__ == "__main__":
    main()
