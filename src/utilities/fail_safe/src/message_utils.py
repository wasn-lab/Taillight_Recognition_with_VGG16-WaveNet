# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Return the corresponding class for a given msg_name
"""
from std_msgs.msg import Bool, Empty, Float64, Int32
from msgs.msg import DetectedObjectArray, VehInfo, BackendInfo
from geometry_msgs.msg import PoseStamped

MSG_TO_CLASS = {
    "BackendInfo": BackendInfo,
    "Bool": Bool,
    "DetectedObjectArray": DetectedObjectArray,
    "PoseStamped": PoseStamped,
    "Empty": Empty,
    "Float64": Float64,
    "Int32": Int32,
    "VehInfo": VehInfo}

def get_message_type_by_str(msg_name):
    """Return the corresponding message type"""
    if msg_name in MSG_TO_CLASS:
        return MSG_TO_CLASS[msg_name]
    raise ValueError("{}: Cannot map to a message type.".format(msg_name))
