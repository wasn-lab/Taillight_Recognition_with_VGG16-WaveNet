#! /usr/bin/env python
import rospy
from msgs.msg import Keypoints
from msgs.msg import Keypoint
from msgs.srv import *

predict_frames = 4
gain = 5  # gain for Kalman Filter

def keypoint_is_detected(keypoint):
    if keypoint.x == 0 and keypoint.y == 0:
        return False
    else:
        return True

def callback(req):
    global predict_frames, gain
    res = []
    
    diff = Keypoints()
    if not req.last_detected_keypoints.keypoint or not req.new_detected_keypoints.keypoint:
        if req.last_detected_keypoints.keypoint:
            for obj in req.last_detected_keypoints.keypoint:
                diff.keypoint.append(obj)
            i = 0
            for i in range(predict_frames):
                res.append(diff)
            return PredictSkeletonResponse(res)
        elif req.new_detected_keypoints.keypoint:
            for obj in req.new_detected_keypoints.keypoint:
                diff.keypoint.append(obj)
            i = 0
            for i in range(predict_frames):
                res.append(diff)
            return PredictSkeletonResponse(res)
        else:
            print("no keypoint detected")
            return PredictSkeletonResponse(res)
    else:  # both last_detected_keypoints and new_detected_keypoints available
        # calculate difference of new_detected_keypoints and last_detected_keypoints
        i = 0
        for i in range(len(req.new_detected_keypoints.keypoint)):
            if keypoint_is_detected(req.last_detected_keypoints.keypoint[i]) and keypoint_is_detected(req.new_detected_keypoints.keypoint[i]):
                diff_keypoint = Keypoint()
                diff_keypoint.x = (req.new_detected_keypoints.keypoint[i].x - req.last_detected_keypoints.keypoint[i].x) / gain
                diff_keypoint.y = (req.new_detected_keypoints.keypoint[i].y - req.last_detected_keypoints.keypoint[i].y) / gain
                diff.keypoint.append(diff_keypoint)
            else:
                diff_keypoint = Keypoint()
                diff_keypoint.x = 0
                diff_keypoint.y = 0
                diff.keypoint.append(diff_keypoint)
        # Predict next N frame's keypoints
        i = 0
        for i in range(predict_frames):
            predict_keypoints = Keypoints()
            j = 0
            for j in range(len(req.new_detected_keypoints.keypoint)):
                predict_keypoint = Keypoint()
                predict_keypoint.x = req.new_detected_keypoints.keypoint[j].x + diff.keypoint[j].x * (i + 1)
                predict_keypoint.y = req.new_detected_keypoints.keypoint[j].y + diff.keypoint[j].y * (i + 1)
                predict_keypoints.keypoint.append(predict_keypoint)
            res.append(predict_keypoints)
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
