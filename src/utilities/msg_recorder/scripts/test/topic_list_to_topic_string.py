#!/usr/bin/env python

f_path = "./"
f_name_topics = "record_topics.txt"

# Read topic_list file
#------------------------#
topic_list_original = []
with open( (f_path+f_name_topics),'r') as _f:
    for _s in _f:
        # Remove the space and '\n'
        _s1 = _s.rstrip().lstrip()
        # Deal with coments
        _idx_comment = _s1.find('#')
        if _idx_comment >= 0: # Do find a '#'
            _s1 = _s1[:_idx_comment].rstrip() # Remove the comment parts
        if len(_s1) > 0: # Append non-empty string (after stripping)
            topic_list_original.append(_s1)
    #
#
# Get unique items (remove duplicated items) and sort
topic_list = sorted(set(topic_list_original))
# print(type(topic_list))
#------------------------#

# Count for duplicated elements
num_duplicated_topic = len(topic_list_original) - len(topic_list)
if num_duplicated_topic > 0:
    # Let's check which topics are duplicated
    __unique_topic_list = list()
    duplicated_topic_list = list()
    for _tp in topic_list_original:
        if not _tp in __unique_topic_list:
            __unique_topic_list.append(_tp)
        else:
            duplicated_topic_list.append(_tp)
    del __unique_topic_list
    duplicated_topic_list = sorted(set(duplicated_topic_list))

# Print the params
# print("param_dict = %s" % str(param_dict))
print("\n\ntopic_list:\n---------------" )
for _tp in topic_list:
    print(_tp)
print("---------------\nNote: Removed %d duplicated topics." % num_duplicated_topic)
if num_duplicated_topic > 0:
    print("\nDuplicated topics:\n---------------")
    for _tp in duplicated_topic_list:
        print(_tp)
print("---------------\n\n" )


#---------------------#
topic_str = " ".join(topic_list)

print("---")
print(topic_str)
print("---")

with open( (f_path+f_name_topics[:-4] + "_str.txt"),'w') as _f:
    _f.write(topic_str)
