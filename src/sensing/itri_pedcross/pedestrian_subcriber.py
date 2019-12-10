#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from msgs.msg import PedObjectArray
from msgs.msg import PedObject
from visualization_msgs.msg import MarkerArray
from pedestrian_marker import pedestrian_marker_callback
from pedestrian_marker import pedestrian_marker_callback_test
from pedestrian_marker import pedestrian_marker_callback_final

def callback(data):
    for element in data.objects:
        print(element)
    print('*')
    
def listener():
    rospy.init_node('node_name')
    rospy.Subscriber('/PathPredictionOutput/lidar/id', MarkerArray, pedestrian_marker_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def listener_test():
    rospy.init_node('node_name')
    rospy.Subscriber('/CamObjFrontCenter', DetectedObjectArray, pedestrian_marker_callback_test)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def listener_pedestrian():
    rospy.init_node('node_name')
    rospy.Subscriber('/PedestrianIntention', PedObjectArray, pedestrian_marker_callback_final)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener_pedestrian()
    except rospy.ROSInterruptException:
        pass
