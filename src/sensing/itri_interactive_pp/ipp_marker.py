#! /usr/bin/python2.7
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
    if input_source == 1:
        marker.header.frame_id = '/map' # vehicle center
    elif input_source == 2:
        marker.header.frame_id = '/base_link' # vehicle center
    else:
        print("Source not found")
        return 0
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
    # print("x: ",position.x)
    # print("y: ",position.y)
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
    pub = rospy.Publisher('/IPP/Marker', MarkerArray, queue_size=1) 
    # rospy.init_node('IPP_marker', anonymous=True) # IPP_marker is TOPIC
    markerArray = MarkerArray()
    for obj in data.objects:
    	if obj.track.is_ready_prediction:
            i = 0
            for track_point in obj.track.forecasts:
                # print("Prediction_horizon: ",i)
                point_2 = obj.bPoint.p1
                point_2.x = track_point.position.x
                point_2.y = track_point.position.y
                point_2.z = 0
                markerArray.markers.append(create_marker(text="9", position=point_2, id=obj.track.id*20 + i, color=[1.0,0.2,0.0]))
                i = i + 1
    #the correct one
    pub.publish(markerArray)


def listener_pedestrian():
    rospy.init_node('IPP_marker', anonymous=True) # IPP_marker is TOPIC
    rospy.Subscriber('/IPP/Alert', DetectedObjectArray, vehicle_marker_callback_final)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    global input_source
    input_source = rospy.get_param('/object_marker/coordinate_type')
    try:
        listener_pedestrian()
    except rospy.ROSInterruptException:
        pass

