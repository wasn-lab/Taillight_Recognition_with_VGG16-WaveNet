#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from msgs.msg import PedObjectArray
from msgs.msg import PedObject
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray

image_buffer = []
crop_image_buffer = []

def hrnet_image_crop_callback(data):
    global crop_image_buffer;
    crop_image_buffer.append(data)
    #print("crop image: ", len(crop_image_buffer))

def hrnet_image_callback(data):
    global image_buffer;
    image_buffer.append(data)
    #print("image: ", len(image_buffer))

def hrnet_objects_callback(data):
    #print("get obj")

def listener_hrnet():
    rospy.init_node('hrnet')
    rospy.Subscriber('/Tracking2D', DetectedObjectArray, hrnet_objects_callback)
    rospy.Subscriber('/cam/front_bottom_60', Image, hrnet_image_callback)
    rospy.Subscriber('/cam/front_bottom_60_crop', Image, hrnet_image_crop_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    #image_buffer
    #crop_image_buffer
    try:
        listener_hrnet()
    except rospy.ROSInterruptException:
        pass
