#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import json

def callback(data):
    json_str = data.data
    # print("json_str = %s" % json_str)
    data_dict = json.loads(json_str)
    print("---\ndata_dict = %s" % str(json.dumps(data_dict, indent=4)))


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('json_through_topic', anonymous=True)

    rospy.Subscriber("/GUI/topic_fps_out", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
