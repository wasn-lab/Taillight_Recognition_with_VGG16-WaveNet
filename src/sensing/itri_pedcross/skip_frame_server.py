#! /usr/bin/env python
import rospy
from msgs.msg import Keypoints
from msgs.msg import Keypoint
from msgs.srv import *

def callback(req):
    res = []
    
    result_of_first_frame = Keypoints()
    for obj in req.last_detected_keypoints:
        result_of_first_frame.keypoint.append(obj)
    for obj in req.new_detected_keypoints:
        result_of_first_frame.keypoint.append(obj)
    res.append(result_of_first_frame)
    
    result_of_second_frame = Keypoints()
    for obj in req.last_detected_keypoints:
        result_of_second_frame.keypoint.append(obj)
    for obj in req.new_detected_keypoints:
        result_of_second_frame.keypoint.append(obj)
    res.append(result_of_second_frame)
    return PredictSkeletonResponse(res)

def listener():
    rospy.init_node('skip_frame_server')
    rospy.Service("skip_frame", PredictSkeleton, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
