#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from msgs.msg import PedObjectArray
from msgs.msg import PedObject

def create_marker(text, position, id = 0 ,duration = 0.1, color=[1.0,1.0,1.0]):
    marker = Marker()
    marker.header.frame_id = '/base_link' # vehicle center
    marker.id = id
    marker.type = marker.TEXT_VIEW_FACING
    marker.text = text
    marker.action = marker.ADD
    #marker.scale.x = 0.05
    #marker.scale.y = 0.05
    marker.scale.z = 2
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.lifetime = rospy.Duration(duration)
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = position.x
    marker.pose.position.y = position.y
    marker.pose.position.z = position.z 
    return marker

#for /CamObjFrontCenter
def pedestrian_marker_callback_final(data):
    pub = rospy.Publisher('/PedCross/3D_marker', MarkerArray, queue_size=1) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    markerArray = MarkerArray()
    for element in data.objects:
        #print(element.track.id)
        if element.classId == 1:
            #temp. due to no fusion
            point = element.bPoint.p1
            point.z = point.z + 2
            if element.crossProbability > 0.55: #threshold
                prob = "C(" + get_two_float(element.crossProbability, 2) + ")"
                markerArray.markers.append(create_marker(text=prob,position=point,id=element.track.id,color=[1.0,0.2,0.0]))
                for pp_point in element.track.forecasts:
                    point_2 = element.bPoint.p1
                    point_2.x = pp_point.position.x
                    point_2.y = pp_point.position.y
                    point_2.z = 0
                    markerArray.markers.append(create_marker(text=".",position=point_2,id=point_2.x*point_2.y,color=[1.0,0.2,0.2]))
            else:
                prob = "NC(" + get_two_float(element.crossProbability, 2) + ")"
                markerArray.markers.append(create_marker(text=prob,position=point,id=element.track.id,color=[0.0,0.9,0.4]))
            #the correct one
    pub.publish(markerArray)

def get_two_float(f_str, n):
    f_str = str(f_str)
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]
    return ".".join([a, c])
