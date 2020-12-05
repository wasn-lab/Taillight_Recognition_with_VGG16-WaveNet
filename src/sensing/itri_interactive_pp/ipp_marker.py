#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time
import numpy as np

def init_new_points(id = 0 ,duration = 0.5, color=[1.0,1.0,1.0]):
    marker = Marker()
    marker.header.stamp = rospy.get_rostime()
    marker.header.frame_id = '/base_link' # vehicle center
    marker.id = id
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.scale.x = 0.4
    marker.scale.y = 0.4
    # marker.scale.z = 0.1
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1
    marker.lifetime = rospy.Duration(duration)
    marker.points = []
    
    return marker

def init_new_line(id = 0 ,duration = 0.5, color=[1.0,1.0,1.0]):
    marker = Marker()
    marker.header.stamp = rospy.get_rostime()
    marker.header.frame_id = "/base_link"
    marker.type = marker.LINE_STRIP
    marker.action = marker.ADD
    marker.id = id
    # marker scale
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    # marker color
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1

    marker.lifetime = rospy.Duration(duration)

    return marker

#for /CamobjFrontCenter
def vehicle_marker_callback_final(data):
    pub = rospy.Publisher('/IPP/Marker', MarkerArray, queue_size=1) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    #print(data.header.frame_id)
    markerArray = MarkerArray()
    for obj in data.objects:
    	if obj.track.is_ready_prediction and obj.track.id == 1754:
            i = 0
            line_marker = init_new_line(id = obj.track.id*30 + i, color=[1.0,0.2,0.0])
            for track_point in obj.track.forecasts:
                point_marker = init_new_points(id=obj.track.id*20 + i)
                print("Prediction_horizon: ",i)
                point_2 = Point()
                point_2.x = track_point.position.x
                point_2.y = track_point.position.y
                point_2.z = 0
                point_marker.points.append(point_2)
                line_marker.points.append(point_2)
                markerArray.markers.append(point_marker)
                i = i + 1
            markerArray.markers.append(line_marker)
                
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

