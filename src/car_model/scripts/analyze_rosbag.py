import argparse
import pprint

import rosbag

def _analyze(bag_filename):
    bag = rosbag.Bag(bag_filename)
    topic_info = []
    _info = bag.get_type_and_topic_info()[1]
    for key in _info:
        doc = {"topic": key,
               "msg_type": _info[key].msg_type,
               "message_count": _info[key].message_count}
        topic_info.append(doc)
    total_bytes = 0
    for doc in topic_info:
        topic = doc["topic"]
        nbytes = 0
        for _topic, raw_msg, _t in bag.read_messages(topics=[topic], raw=True):
            _msg_type, serialized_bytes, _md5sum, _pos, _pytype = raw_msg
            nbytes += len(serialized_bytes)
        doc["num_bytes"] = nbytes
        nmsgs = doc["message_count"]
        doc["avg_msg_size_in_bytes"] = float(nbytes) / max(nmsgs, 1)
        total_bytes += nbytes
    pprint.pprint(topic_info)
    print("Total bytes: {}".format(total_bytes))
    print("-" * 30)
    print("Topics that has average message size > 4096:")
    for doc in topic_info:
        if doc["avg_msg_size_in_bytes"] > 4096:
            print(doc["topic"])
    bag.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", required=True)
    args = parser.parse_args()
    _analyze(args.rosbag)

if __name__ == "__main__":
    main()
