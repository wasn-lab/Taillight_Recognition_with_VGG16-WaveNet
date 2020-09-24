#! /usr/bin/env python
import rospy
from msgs.msg import Keypoints
from msgs.msg import Keypoint
from msgs.srv import *
import time

def test():
    rospy.init_node('skip_frame_client')
    start = time.time()
    rospy.wait_for_service("skip_frame")
    try:
        skip_frame_client = rospy.ServiceProxy("skip_frame", PredictSkeleton)
        req = Keypoints()
        keypoint = Keypoint()
        keypoint.x = 1
        keypoint.y = 2
        keypoint2 = Keypoint()
        keypoint2.x = 3
        keypoint2.y = 4
        # append 25 keypoints for testing
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        req.keypoint.append(keypoint2)
        req.keypoint.append(keypoint)
        # call server (last_detected_keypoints, new_detected_keypoints)
        start = time.time()
        res = skip_frame_client.call(req, req)
        print(res)
    except rospy.ServiceException:
        rospy.logwarn("Service call failed.")
    stop = time.time()
    print("latency: " + str(stop - start))

if __name__ == '__main__':
    try:
        test()
    except rospy.ROSInterruptException:
        pass
