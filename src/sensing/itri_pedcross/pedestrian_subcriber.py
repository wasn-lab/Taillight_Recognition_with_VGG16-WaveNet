#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from msgs.msg import PedObjectArray
from msgs.msg import PedObject
from visualization_msgs.msg import MarkerArray
from pedestrian_marker import pedestrian_marker_callback_final

def listener_pedestrian():
    rospy.init_node('node_name')
    rospy.Subscriber('/PedCross/Pedestrians/front_bottom_60', PedObjectArray, pedestrian_marker_callback_final)
    rospy.Subscriber('/PedCross/Pedestrians/front_top_far_30', PedObjectArray, pedestrian_marker_callback_final)
    rospy.Subscriber('/PedCross/Pedestrians/left_back_60', PedObjectArray, pedestrian_marker_callback_final)
    rospy.Subscriber('/PedCross/Pedestrians/right_back_60', PedObjectArray, pedestrian_marker_callback_final)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener_pedestrian()
    except rospy.ROSInterruptException:
        pass
