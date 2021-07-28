#! /usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

rospy.init_node('rviz_marker')

marker_pub = rospy.Publisher("/my_marker", MarkerArray, queue_size = 2)

lg_marker = Marker()
sg_marker = Marker()
rg_marker = Marker()
r_marker = Marker()
y_marker = Marker()
lampole = Marker()
countdowan_text = Marker()

markerArray = MarkerArray()

lg_marker.header.frame_id = "/map"
lg_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
lg_marker.type = 0
lg_marker.id = 3

# Set the scale of the marker
lg_marker.scale.x = 0.3
lg_marker.scale.y = 0.5
lg_marker.scale.z = 0.0

# Set the color
lg_marker.color.r = 0.0
lg_marker.color.g = 1.0
lg_marker.color.b = 0.0
lg_marker.color.a = 1.0

# Set the pose of the marker
#g_marker.pose.position.x = 2.7
#g_marker.pose.position.y = 0
#g_marker.pose.position.z = 3
tail_l = Point(2.5, 0, 10.0)
tip_l = Point(1.5, 0, 10.0)
lg_marker.points = [tail_l, tip_l]
lg_marker.pose.orientation.x = 0.0
lg_marker.pose.orientation.y = 0.0
lg_marker.pose.orientation.z = 0.0
lg_marker.pose.orientation.w = 1.0





sg_marker.header.frame_id = "/map"
sg_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
sg_marker.type = 0
sg_marker.id = 4

# Set the scale of the marker
sg_marker.scale.x = 0.3
sg_marker.scale.y = 0.5
sg_marker.scale.z = 0.0

# Set the color
sg_marker.color.r = 0.0
sg_marker.color.g = 1.0
sg_marker.color.b = 0.0
sg_marker.color.a = 1.0

# Set the pose of the marker
#g_marker.pose.position.x = 2.7
#g_marker.pose.position.y = 0
#g_marker.pose.position.z = 3
tail_s = Point(3.0, 0, 9.6)
tip_s = Point(3.0, 0, 10.4)
sg_marker.points = [tail_s, tip_s]
sg_marker.pose.orientation.x = 0.0
sg_marker.pose.orientation.y = 0.0
sg_marker.pose.orientation.z = 0.0
sg_marker.pose.orientation.w = 1.0





rg_marker.header.frame_id = "/map"
rg_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
rg_marker.type = 0
rg_marker.id = 5

# Set the scale of the marker
rg_marker.scale.x = 0.3
rg_marker.scale.y = 0.5
rg_marker.scale.z = 0.0

# Set the color
rg_marker.color.r = 0.0
rg_marker.color.g = 1.0
rg_marker.color.b = 0.0
rg_marker.color.a = 1.0

# Set the pose of the marker
#g_marker.pose.position.x = 2.7
#g_marker.pose.position.y = 0
#g_marker.pose.position.z = 3
tail_r = Point(3.5, 0, 10.0)
tip_r = Point(4.5, 0, 10.0)
rg_marker.points = [tail_r, tip_r]
rg_marker.pose.orientation.x = 0.0
rg_marker.pose.orientation.y = 0.0
rg_marker.pose.orientation.z = 0.0
rg_marker.pose.orientation.w = 1.0





r_marker.header.frame_id = "/map"
r_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
r_marker.type = 2
r_marker.id = 1

# Set the scale of the marker
r_marker.scale.x = 1.0
r_marker.scale.y = 1.0
r_marker.scale.z = 1.0

# Set the color
r_marker.color.r = 1.0
r_marker.color.g = 0.0
r_marker.color.b = 0.0
r_marker.color.a = 1.0

# Set the pose of the marker
r_marker.pose.position.x = 0
r_marker.pose.position.y = 0
r_marker.pose.position.z = 10
r_marker.pose.orientation.x = 0.0
r_marker.pose.orientation.y = 0.0
r_marker.pose.orientation.z = 0.0
r_marker.pose.orientation.w = 1.0



y_marker.header.frame_id = "/map"
y_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
y_marker.type = 2
y_marker.id = 2

# Set the scale of the marker
y_marker.scale.x = 1.0
y_marker.scale.y = 1.0
y_marker.scale.z = 1.0

# Set the color
y_marker.color.r = 1.0
y_marker.color.g = 1.0
y_marker.color.b = 0.0
y_marker.color.a = 1.0

# Set the pose of the marker
y_marker.pose.position.x = 1
y_marker.pose.position.y = 0
y_marker.pose.position.z = 10
y_marker.pose.orientation.x = 0.0
y_marker.pose.orientation.y = 0.0
y_marker.pose.orientation.z = 0.0
y_marker.pose.orientation.w = 1.0







y_marker.header.frame_id = "/map"
y_marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
y_marker.type = 2
y_marker.id = 2

# Set the scale of the marker
y_marker.scale.x = 1.0
y_marker.scale.y = 1.0
y_marker.scale.z = 1.0

# Set the color
y_marker.color.r = 1.0
y_marker.color.g = 1.0
y_marker.color.b = 0.0
y_marker.color.a = 1.0

# Set the pose of the marker
y_marker.pose.position.x = 1
y_marker.pose.position.y = 0
y_marker.pose.position.z = 10
y_marker.pose.orientation.x = 0.0
y_marker.pose.orientation.y = 0.0
y_marker.pose.orientation.z = 0.0
y_marker.pose.orientation.w = 1.0







lampole.header.frame_id = "/map"
lampole.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
lampole.type = 3
lampole.id = 6

# Set the scale of the marker
lampole.scale.x = 0.3
lampole.scale.y = 0.3
lampole.scale.z = 10.0

# Set the color
lampole.color.r = 0.7
lampole.color.g = 0.7
lampole.color.b = 0.7
lampole.color.a = 1.0

# Set the pose of the marker

lampole.pose.position.x = -0.7
lampole.pose.position.y = 0
lampole.pose.position.z = 5

lampole.pose.orientation.x = 0.0
lampole.pose.orientation.y = 0.0
lampole.pose.orientation.z = 0.0
lampole.pose.orientation.w = 1.0




countdowan_text.header.frame_id = "/map"
countdowan_text.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
countdowan_text.type = 9
countdowan_text.id = 7

# Set the scale of the marker

countdowan_text.scale.z = 1.0
countdowan_text.text = "99"

# Set the color
countdowan_text.color.r = 1.0
countdowan_text.color.g = 1.0
countdowan_text.color.b = 1.0
countdowan_text.color.a = 1.0

# Set the pose of the marker
countdowan_text.pose.position.x = 5.1
countdowan_text.pose.position.y = 0
countdowan_text.pose.position.z = 10

countdowan_text.pose.orientation.x = 0.0
countdowan_text.pose.orientation.y = 0.0
countdowan_text.pose.orientation.z = 0.0
countdowan_text.pose.orientation.w = 1.0





markerArray.markers.append(lg_marker)
markerArray.markers.append(sg_marker)
markerArray.markers.append(rg_marker)
markerArray.markers.append(r_marker)
markerArray.markers.append(y_marker)
markerArray.markers.append(lampole)
markerArray.markers.append(countdowan_text)



while not rospy.is_shutdown():
  marker_pub.publish(markerArray)
  rospy.rostime.wallsleep(1.0)

