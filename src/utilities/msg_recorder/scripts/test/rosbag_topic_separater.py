import rosbag
import os

#-----------------#
input_bag_name = "input.bag"
output_bag_dir = "sep_out/"
#-----------------#

# Creating directories
try:
    os.makedirs(output_bag_dir)
    print("The directory <%s> has been created." % output_bag_dir)
except:
    print("The directry <%s> already exists." % output_bag_dir)
#

#--------------------------------#
bag = rosbag.Bag(input_bag_name)
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
    types.append(bag.get_type_and_topic_info()[1].values()[i][0])

# print("bag.get_type_and_topic_info() = %s" % str(bag.get_type_and_topic_info()))
print("topics = %s" % str(topics))
print("types = %s" % str(types))


#--------------------------------#
topic_to_bag_name_dict = dict()
# for _topic in topics:
#     topic_to_bag_name_dict[_topic] = _topic.replace('/', '@') + ".bag"
# print("topic_to_bag_name_dict = %s" % str(topic_to_bag_name_dict))

msg_count = 0

print('-'*70)
print("Start looping!!")
print('-'*70)
# Loop the messages of input bag
for _topic, msg, t in rosbag.Bag(input_bag_name).read_messages():
    msg_count += 1
    if not _topic in topic_to_bag_name_dict:
        topic_to_bag_name_dict[_topic] = _topic.replace('/', '@') + ".bag"
    #
    print("#%d\tWrite file <%s>" % (msg_count, topic_to_bag_name_dict[_topic]))
    _file_path = output_bag_dir + topic_to_bag_name_dict[_topic]
    try:
        with rosbag.Bag(_file_path, 'a') as outbag:
            outbag.write(_topic, msg, t)
    except:
        with rosbag.Bag(_file_path, 'w') as outbag:
            outbag.write(_topic, msg, t)

print('-'*70)
print("Finished!!")
print('-'*70)
print("topic_to_bag_name_dict = %s" % str(topic_to_bag_name_dict))
