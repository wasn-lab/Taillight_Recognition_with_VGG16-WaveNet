from std_msgs.msg import Bool, Empty, Int32
from msgs.msg import DetectedObjectArray, VehInfo

def get_message_type_by_str(msg_name):
    if msg_name == "Empty":
        return Empty
    if msg_name == "Int32":
        return Int32
    if msg_name == "Bool":
        return Bool
    if msg_name == "DetectedObjectArray":
        return DetectedObjectArray
    if msg_name == "VehInfo":
        return VehInfo
