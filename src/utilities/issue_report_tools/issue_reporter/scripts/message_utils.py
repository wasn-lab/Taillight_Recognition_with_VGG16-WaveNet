from std_msgs.msg import Empty, Int32

def get_message_type_by_str(msg_name):
    if msg_name == "Empty":
        return Empty
    if msg_name == "Int32":
        return Int32
