import rosbag
bag = rosbag.Bag('input.bag')
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
    types.append(bag.get_type_and_topic_info()[1].values()[i][0])

print("bag.get_type_and_topic_info() = %s" % str(bag.get_type_and_topic_info()))

print("topics = %s" % str(topics))
print("types = %s" % str(types))
