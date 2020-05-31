import os
import glob
import csv


# Get the list of bag name
bag_name_list = sorted( glob.glob("*.bag") )

#
bag_name_to_size_dict = dict()
bag_name_to_percent_dict = dict() # size in percentage
bag_size_name_tuple_list = list()
#

# Display the list of bag name
# Statistic about size
total_size_in_MB = 0.0
topic_count = 0
for idx, name in enumerate(bag_name_list):
    # print(os.stat(name))
    size_in_byte = os.stat(name).st_size
    size_in_MB = size_in_byte/float(1024**2)
    total_size_in_MB += size_in_MB
    #
    bag_name_to_size_dict[name] = size_in_MB
    bag_size_name_tuple_list.append( (size_in_MB, name) )
    # print("%d: %s\t size=%f MB" % (idx+1, name, size_in_MB))

topic_count = len(bag_size_name_tuple_list)

print("Total size = %f, with %d topics" % (total_size_in_MB, topic_count))

# Calculate percentage
for name in bag_name_to_size_dict:
    bag_name_to_percent_dict[name] = bag_name_to_size_dict[name]/total_size_in_MB

#-----------------------#
# Sort by file size
bag_size_name_tuple_list.sort(reverse=True) # Larger one go first

# list of dict
list_bag_info = list()

# List the sorted statistic result
for idx, _pack in enumerate(bag_size_name_tuple_list):
    size_in_MB, name = _pack
    print("%d:\tsize=%f MB\t(%f)\t%s" % (idx+1, size_in_MB, bag_name_to_percent_dict[name], name))
    #
    _d = dict()
    _d["ranking"] = idx + 1
    _d["size(MB)"] = size_in_MB
    _d["ratio in file"] = bag_name_to_percent_dict[name]
    _d["topic name"] = name.replace("@", "/").replace(".bag", "")
    list_bag_info.append(_d)



# Write the result into a CSV file
try:
    with open('./topic_size.csv', mode='w') as csv_file:
        fieldnames = list(list_bag_info[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for _d in list_bag_info:
            writer.writerow(_d)
    print("Done writing CSV")
except:
    print("Error writing CSV")
