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

def pedestrian_marker():
    pub = rospy.Publisher('/PedCross/3D_marker', MarkerArray, queue_size=10) # pedestrian_marker is TOPIC
    rospy.init_node('pedestrian_marker', anonymous=True)
    markerArray = MarkerArray()
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        markerArray.markers = []
        markerArray.markers.append(create_marker(text="C",position=[0,0,0],id=0))
        markerArray.markers.append(create_marker(text="NC",position=[5,5,5],id=1))
        pub.publish(markerArray)
        rate.sleep()

#for /PathPredictionOutput
def pedestrian_marker_callback(data):
    pub = rospy.Publisher('pedestrian_marker', MarkerArray, queue_size=10) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    markerArray = MarkerArray()
    #rate = rospy.Rate(10) # 10hz
    for element in data.markers:
        #markerArray.markers = []
        markerArray.markers.append(create_marker(text=str(element.type),position=element.pose.position,id=element.id))
        #markerArray.markers.append(create_marker(text="NC",position=[5,5,5],id=1))
    pub.publish(markerArray)
    #rate.sleep()

#for /CamObjFrontCenter
def pedestrian_marker_callback_test(data):
    pub = rospy.Publisher('pedestrian_marker', MarkerArray, queue_size=10) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    markerArray = MarkerArray()
    count = 0
    for element in data.objects:
        #print(element.fusionSourceId)
        if element.classId == 1:
            #temp. due to no fusion
            markerArray.markers.append(create_marker(text='C 100%',position=element.bPoint.p0,id=count))
            #the correct one
            #markerArray.markers.append(create_marker(text='C 100%',position=element.bPoint.p0,id=element.fusionSourceId))
            count = count + 1
        
    pub.publish(markerArray)

#for /CamObjFrontCenter
def pedestrian_marker_callback_final(data):
    pub = rospy.Publisher('/PedCross/3D_marker', MarkerArray, queue_size=1) # pedestrian_marker is TOPIC
    #rospy.init_node('pedestrian_marker', anonymous=True)
    markerArray = MarkerArray()
    count = 0
    for element in data.objects:
        #print(element.fusionSourceId)
        if element.classId == 1:
            #temp. due to no fusion
            prob = "pc="+str(int(element.crossProbability*100))+"%";
            if element.crossProbability > 0.5: #threshold
                prob = "C "+prob
            else:
                prob = "NC "+prob
            markerArray.markers.append(create_marker(text=prob,position=element.bPoint.p0,id=element.track.id))
            #the correct one
            #markerArray.markers.append(create_marker(text='C 100%',position=element.bPoint.p0,id=element.fusionSourceId))
            count = count + 1
        
    pub.publish(markerArray)
    

if __name__ == '__main__':
    try:
        pedestrian_marker()
    except rospy.ROSInterruptException:
        pass
