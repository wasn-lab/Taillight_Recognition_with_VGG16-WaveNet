#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time
import numpy as np

def create_marker(text, position, id = 0 ,duration = 0.5, color=[1.0,1.0,1.0]):
    marker = Marker()
    marker.header.stamp = rospy.get_rostime()
    marker.header.frame_id = '/base_link' # vehicle center
    marker.id = id
    marker.type = marker.TEXT_VIEW_FACING
    marker.text = text
    marker.action = marker.ADD
    #marker.scale.x = 0.05
    #marker.scale.y = 0.05
    marker.scale.z = 2
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1
    marker.lifetime = rospy.Duration(duration)
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = position.x
    marker.pose.position.y = position.y
    marker.pose.position.z = position.z
    print("x: ",position.x)
    print("y: ",position.y)
    return marker

# def create_line():
#     marker = Marker()
#     marker.header.frame_id = "/base_link"
#     marker.type = marker.LINE_STRIP
#     marker.action = marker.ADD

#     # marker scale
#     marker.scale.x = 0.03
#     marker.scale.y = 0.03
#     marker.scale.z = 0.03

#     # marker color
#     marker.color.a = 1.0
#     marker.color.r = 1.0
#     marker.color.g = 1.0
#     marker.color.b = 0.0

#     # marker orientaiton
#     marker.pose.orientation.x = 0.0
#     marker.pose.orientation.y = 0.0
#     marker.pose.orientation.z = 0.0
#     marker.pose.orientation.w = 1.0

#     # marker position
#     marker.pose.position.x = 0.0
#     marker.pose.position.y = 0.0
#     marker.pose.position.z = 0.0

#     # marker line points
#     marker.points = []
#     # first point
#     first_line_point = Point()
#     first_line_point.x = 0.0
#     first_line_point.y = 0.0
#     first_line_point.z = 0.0
#     marker.points.append(first_line_point)
#     # second point
#     second_line_point = Point()
#     second_line_point.x = 1.0
#     second_line_point.y = 1.0
#     second_line_point.z = 0.0

#for /CamelementFrontCenter
def vehicle_marker_callback_final(data):
    pub = rospy.Publisher('/IPP/Marker', MarkerArray, queue_size=1) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    #print(data.header.frame_id)
    markerArray = MarkerArray()
    for element in data.objects:

    	if element.track.is_ready_prediction and element.track.id == 1754:
            print(element.track.id)
            x =(element.bPoint.p0.x + element.bPoint.p1.x + element.bPoint.p2.x + element.bPoint.p3.x + element.bPoint.p4.x + element.bPoint.p5.x + element.bPoint.p6.x + element.bPoint.p7.x) / 8
            y =(element.bPoint.p0.y + element.bPoint.p1.y + element.bPoint.p2.y + element.bPoint.p3.y + element.bPoint.p4.y + element.bPoint.p5.y + element.bPoint.p6.y + element.bPoint.p7.y) / 8
            print('element.bPoint.p0.x: ',x)
            print('element.bPoint.p0.y: ',y)
            i = 0
            for track_point in element.track.forecasts:
                print("Prediction_horizon: ",i)
                point_2 = element.bPoint.p1
                point_2.x = track_point.position.x
                point_2.y = track_point.position.y
                point_2.z = 0
                markerArray.markers.append(create_marker(text="9", position=point_2, id=element.track.id*20 + i, color=[1.0,0.2,0.0]))
                i = i + 1
    #the correct one
    pub.publish(markerArray)


def listener_pedestrian():
    rospy.init_node('node_name')
    rospy.Subscriber('/IPP/Alert', DetectedObjectArray, vehicle_marker_callback_final)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener_pedestrian()
    except rospy.ROSInterruptException:
        pass

