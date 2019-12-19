import yaml
import json

bag_dict_list = []

with open("./backup_history.txt") as _F:
    bag_dict = yaml.load_all(_F)
    print(bag_dict)
    for _d in bag_dict:
        bag_dict_list.append(_d)

print(bag_dict_list)


for _d in bag_dict_list:
    print("Triggered at: %s" % str(_d["timestamp"]))
    print("reason: %s" % _d["reason"])
    print("bags:")
    for _bag in _d["bags"]: # a list
        print("- %s" % _bag)
    print("\n")
