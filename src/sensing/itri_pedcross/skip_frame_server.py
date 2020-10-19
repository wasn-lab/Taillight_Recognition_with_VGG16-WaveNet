#! /usr/bin/env python
import rospy
from msgs.msg import Keypoints
from msgs.msg import Keypoint
from msgs.srv import *

predict_frames = 1  # frame number for skip frame
gain = 5  # gain for Kalman Filter
frame_num = 10  # frame number for crossing predictionz
do_back_prediction = True

def keypoint_is_detected(keypoint):
    if keypoint.x == 0 and keypoint.y == 0:
        return False
    else:
        return True

def callback(req):
    global predict_frames, gain, do_back_prediction
    predicted_keypoints = []
    processed_keypoints = []

    processed_keypoints = req.original_keypoints

    if predict_frames == 0:
        return PredictSkeletonResponse(predicted_keypoints, processed_keypoints)
    
    latest_openpose = req.original_keypoints[frame_num - 1]
    second_latest_openpose = req.original_keypoints[frame_num - 2 - predict_frames]
    # calculate predicted_keypoints
    diff = Keypoints()
    for i in range(len(latest_openpose.keypoint)):
        diff_keypoint = Keypoint()
        if keypoint_is_detected(latest_openpose.keypoint[i]) and keypoint_is_detected(second_latest_openpose.keypoint[i]):
            diff_keypoint.x = (latest_openpose.keypoint[i].x - second_latest_openpose.keypoint[i].x) / gain
            diff_keypoint.y = (latest_openpose.keypoint[i].y - second_latest_openpose.keypoint[i].y) / gain
        else:
            diff_keypoint.x = 0
            diff_keypoint.y = 0
        diff.keypoint.append(diff_keypoint)
    for i in range(predict_frames):
        predict_keypoints = Keypoints()
        for j in range(len(diff.keypoint)):
            predict_keypoint = Keypoint()
            predict_keypoint.x = latest_openpose.keypoint[j].x + diff.keypoint[j].x * (i + 1)
            predict_keypoint.y = latest_openpose.keypoint[j].y + diff.keypoint[j].y * (i + 1)
            predict_keypoints.keypoint.append(predict_keypoint)
        predicted_keypoints.append(predict_keypoints)
    
    if do_back_prediction:
        # calculate processed_keypoints
        diff = Keypoints()
        for i in range(len(latest_openpose.keypoint)):
            diff_keypoint = Keypoint()
            if keypoint_is_detected(latest_openpose.keypoint[i]) and keypoint_is_detected(second_latest_openpose.keypoint[i]):
                diff_keypoint.x = (latest_openpose.keypoint[i].x - second_latest_openpose.keypoint[i].x) / (predict_frames + 1)
                diff_keypoint.y = (latest_openpose.keypoint[i].y - second_latest_openpose.keypoint[i].y) / (predict_frames + 1)
            else:
                diff_keypoint.x = 0
                diff_keypoint.y = 0
            diff.keypoint.append(diff_keypoint)
        for i in range(predict_frames):
            predict_keypoints = Keypoints()
            for j in range(len(diff.keypoint)):
                if keypoint_is_detected(diff.keypoint[j]):
                    # processed_keypoints = (original keypoints + interpolation keypoints) / 2
                    processed_keypoints[frame_num - 2 - predict_frames + (i + 1)].keypoint[j].x = (processed_keypoints[frame_num - 2 - predict_frames + (i + 1)].keypoint[j].x + second_latest_openpose.keypoint[j].x + diff.keypoint[j].x * (i + 1)) / 2
                    processed_keypoints[frame_num - 2 - predict_frames + (i + 1)].keypoint[j].y = (processed_keypoints[frame_num - 2 - predict_frames + (i + 1)].keypoint[j].y + second_latest_openpose.keypoint[j].y + diff.keypoint[j].y * (i + 1)) / 2

    return PredictSkeletonResponse(predicted_keypoints, processed_keypoints)

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
