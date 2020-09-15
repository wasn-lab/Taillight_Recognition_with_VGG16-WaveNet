from std_msgs.msg import Empty

def get_message_type_by_str(msg_name):
    if msg_name == "Empty":
        return Empty
