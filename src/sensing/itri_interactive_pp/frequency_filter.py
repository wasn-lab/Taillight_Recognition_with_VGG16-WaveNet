import rospy
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import PoseStamped
from msgs.msg import DetectedObjectArray

import time

prev = rospy.Time()
count = 0
past_obj = []
tf_buffer = None
tf_listener = None

def publish_msg(data):
    global prev, count
    print('_')
    if(data.header.stamp - prev).to_sec() > 10:
        print('Initialize')
    if(data.header.stamp - prev).to_sec() > 0.5:
        print((data.header.stamp - prev).to_sec())
        prev = data.header.stamp
        pub = rospy.Publisher('/IPP/delay_Alert', DetectedObjectArray, queue_size=1) # /IPP/Alert is TOPIC
        pub.publish(data)
        count = 0
    else:
        count = count + 1
        
    if count >= 10:
        print('Restart')
        prev = rospy.Time()


def listener_ipp():
    global tf_buffer, tf_listener
    rospy.init_node('ipp_delay_data')
    rospy.Subscriber('/Tracking2D/front_bottom_60', DetectedObjectArray, publish_msg)
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        listener_ipp()
    except rospy.ROSInterruptException:
        pass