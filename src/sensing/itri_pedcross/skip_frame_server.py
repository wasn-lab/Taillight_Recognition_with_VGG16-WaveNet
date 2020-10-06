#! /usr/bin/env python
import rospy
from msgs.msg import Keypoints
from msgs.msg import Keypoint
from msgs.srv import *

predict_frames = 1
gain = 5  # gain for Kalman Filter

def keypoint_is_detected(keypoint):
    if keypoint.x == 0 and keypoint.y == 0:
        return False
    else:
        return True

def callback(req):
    global predict_frames, gain
    predicted_keypoints = []
    back_predicted_keypoints = []
    
    diff = Keypoints()
    if not req.previous_one_keypoints.keypoint or not req.new_detected_keypoints.keypoint:
        if req.previous_one_keypoints.keypoint:
            for obj in req.previous_one_keypoints.keypoint:
                diff.keypoint.append(obj)
            i = 0
            for i in range(predict_frames):
                predicted_keypoints.append(diff)
            return PredictSkeletonResponse(predicted_keypoints, back_predicted_keypoints)
        elif req.new_detected_keypoints.keypoint:
            for obj in req.new_detected_keypoints.keypoint:
                diff.keypoint.append(obj)
            i = 0
            for i in range(predict_frames):
                predicted_keypoints.append(diff)
            return PredictSkeletonResponse(predicted_keypoints, back_predicted_keypoints)
        else:
            print("no keypoint detected")
            return PredictSkeletonResponse(predicted_keypoints, back_predicted_keypoints)
    else:  # both previous_one_keypoints and new_detected_keypoints available
        # calculate difference of new_detected_keypoints and previous_one_keypoints
        i = 0
        for i in range(len(req.new_detected_keypoints.keypoint)):
            if keypoint_is_detected(req.previous_one_keypoints.keypoint[i]) and keypoint_is_detected(req.new_detected_keypoints.keypoint[i]):
                diff_keypoint = Keypoint()
                diff_keypoint.x = (req.new_detected_keypoints.keypoint[i].x - req.previous_one_keypoints.keypoint[i].x) / gain
                diff_keypoint.y = (req.new_detected_keypoints.keypoint[i].y - req.previous_one_keypoints.keypoint[i].y) / gain
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
            predicted_keypoints.append(predict_keypoints)

        # Back prediction, need previous_two_keypoints
        if req.previous_two_keypoints.keypoint:
            diff_interpolation = Keypoints()
            diff_extrapolation = Keypoints()
            # Calculate difference between previous_two_keypoints and previous_one_keypoints (extrapolation)
            i = 0
            for i in range(len(req.previous_one_keypoints.keypoint)):
                if keypoint_is_detected(req.previous_two_keypoints.keypoint[i]) and keypoint_is_detected(req.previous_one_keypoints.keypoint[i]):
                    diff_keypoint = Keypoint()
                    diff_keypoint.x = (req.previous_one_keypoints.keypoint[i].x - req.previous_two_keypoints.keypoint[i].x) / gain
                    diff_keypoint.y = (req.previous_one_keypoints.keypoint[i].y - req.previous_two_keypoints.keypoint[i].y) / gain
                    diff_extrapolation.keypoint.append(diff_keypoint)
                else:
                    diff_keypoint = Keypoint()
                    diff_keypoint.x = 0
                    diff_keypoint.y = 0
                    diff_extrapolation.keypoint.append(diff_keypoint)
            # Calculate difference between previous_one_keypoints and new_detected_keypoints (interpolation)
            i = 0
            for i in range(len(req.previous_one_keypoints.keypoint)):
                if keypoint_is_detected(req.previous_one_keypoints.keypoint[i]) and keypoint_is_detected(req.new_detected_keypoints.keypoint[i]):
                    diff_keypoint = Keypoint()
                    diff_keypoint.x = (req.new_detected_keypoints.keypoint[i].x - req.previous_one_keypoints.keypoint[i].x) / gain
                    diff_keypoint.y = (req.new_detected_keypoints.keypoint[i].y - req.previous_one_keypoints.keypoint[i].y) / gain
                    diff_interpolation.keypoint.append(diff_keypoint)
                else:
                    diff_keypoint = Keypoint()
                    diff_keypoint.x = 0
                    diff_keypoint.y = 0
                    diff_interpolation.keypoint.append(diff_keypoint)
            i = 0
            for i in range(predict_frames):
                extrapolation_keypoints = Keypoints()
                interpolation_keypoints = Keypoints()
                predict_keypoints = Keypoints()
                # extrapolation
                j = 0
                for j in range(len(req.previous_one_keypoints.keypoint)):
                    predict_keypoint = Keypoint()
                    predict_keypoint.x = req.previous_one_keypoints.keypoint[j].x + diff_extrapolation.keypoint[j].x * (i + 1)
                    predict_keypoint.y = req.previous_one_keypoints.keypoint[j].y + diff_extrapolation.keypoint[j].y * (i + 1)
                    extrapolation_keypoints.keypoint.append(predict_keypoint)
                # interpolation
                j = 0
                for j in range(len(req.previous_one_keypoints.keypoint)):
                    predict_keypoint = Keypoint()
                    predict_keypoint.x = req.previous_one_keypoints.keypoint[j].x + diff_interpolation.keypoint[j].x * (i + 1)
                    predict_keypoint.y = req.previous_one_keypoints.keypoint[j].y + diff_interpolation.keypoint[j].y * (i + 1)
                    interpolation_keypoints.keypoint.append(predict_keypoint)
                # get mean of interpolation and extrapolation
                j = 0
                for j in range(len(extrapolation_keypoints.keypoint)):
                    predict_keypoint = Keypoint()
                    interpolation_keypoints.keypoint[j].x = (interpolation_keypoints.keypoint[j].x + extrapolation_keypoints.keypoint[j].x) / 2
                    interpolation_keypoints.keypoint[j].y = (interpolation_keypoints.keypoint[j].y + extrapolation_keypoints.keypoint[j].y) / 2
                    predict_keypoints.keypoint.append(predict_keypoint)
                back_predicted_keypoints.append(predict_keypoints)

        return PredictSkeletonResponse(predicted_keypoints, back_predicted_keypoints)

def listener():
    global predict_frames
    rospy.init_node('skip_frame_server')
    predict_frames = rospy.get_param('/skip_frame_server/skip_frame_number')
    rospy.Service("skip_frame", PredictSkeleton, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
