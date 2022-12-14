#! ./sandbox/bin/python2.7

import rospy
import tf2_ros
from msgs.msg import DetectedObjectArray
import time


def publish_msg(data):
    global prev
    # , count
    # print('_')
    # if(data.header.stamp - prev).to_sec() > 10:
    # print('Initialize')

    # print (data.header.stamp).to_sec()
    if(data.header.stamp - prev).to_sec() > 0.5:
        # print((data.header.stamp - prev).to_sec())
        # clear the previous prediction result
        for obj in data.objects:
            obj.track.forecasts = []
            obj.track.is_ready_prediction = False
        prev = data.header.stamp
        pub = rospy.Publisher(
            '/IPP/delay_Alert',
            DetectedObjectArray,
            queue_size=1)  # /IPP/Alert is TOPIC
        pub.publish(data)
    #     count = 0
    # else:
    #     count = count + 1

    # if count >= 10:
    #     # print('Restart')
    #     prev = rospy.Time()


def listener_ipp():
    if input_source == 1:
        rospy.Subscriber(
            '/Tracking2D/front_bottom_60',
            DetectedObjectArray,
            publish_msg)
    elif input_source == 2:
        rospy.Subscriber(
            '/PathPredictionOutput',
            DetectedObjectArray,
            publish_msg)
    elif input_source == 3:
        rospy.Subscriber('/Tracking3D', DetectedObjectArray, publish_msg)
    else:
        print("Source not found!")
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
    # spin() simply keeps python from exiting until this node is stopped
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.spin()


if __name__ == '__main__':
    global prev
    # ,count
    # count = 0
    prev = rospy.Time()
    rospy.init_node('ipp_delay_data')
    input_source = rospy.get_param('/filter/input_topic')
    try:
        listener_ipp()
    except rospy.ROSInterruptException:
        pass
