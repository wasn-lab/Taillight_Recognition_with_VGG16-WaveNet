#! ./sandbox/bin/python2.7
# coding=utf-8
import math
import atexit
import rospy
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import PoseStamped
from std_msgs.msg import String
import pandas as pdGIT
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
import time

def distance(point1, point2):
    return math.sqrt(
        math.pow(
            (point1[0] -
             point2[0]),
            2) +
        math.pow(
            (point1[1] -
             point2[1]),
            2))


def record_call_back(data):
    if len(pred_buffer.items())==0:
        TIMER = 0
    else:
        TIMER = max(pred_buffer.keys()) + 1
        
    print("Time : ", TIMER, " Get Data !")
    id_pred = dict()
    id_gt = dict()
    pred = list()
    for obj in data.objects:
        id = int(obj.track.id)
        x_gt = obj.center_point.x
        y_gt = obj.center_point.y
        z_gt = obj.center_point.z
        if tf_map:
            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = x_gt
            pose_stamped.pose.position.y = y_gt
            pose_stamped.pose.position.z = z_gt
            pose_transformed = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform)
            x_gt = pose_transformed.pose.position.x
            y_gt = pose_transformed.pose.position.y
            z_gt = pose_transformed.pose.position.z
        id_gt[id] = [x_gt, y_gt]
        if obj.track.is_ready_prediction:
            for track_point in obj.track.forecasts:
                x_pred = track_point.position.x
                y_pred = track_point.position.y
                pred.append([x_pred, y_pred])
        id_pred[id] = pred
    pred_buffer[TIMER] = id_pred
    gt_buffer[TIMER] = id_gt


def calculate():
    # calculate part
    print('Pred_buffer : ', pred_buffer)
    Timer = max(pred_buffer.keys())
    for time in range(TIMER):
        for id in pred_buffer[time].keys():
            count = 0
            # If our prediction is longer than grountruth ex : bag only 10s we
            # predict 12s result
            time_dist_list = list()
            max_gt_in_pred = min(TIMER - time, 10)
            for i in range(max_gt_in_pred):
                # check for the value is exist or not (If id switch in future
                # the id won't exist)
                if id in gt_buffer[time + i].keys():
                    # print("gt :",gt_buffer[time+i][id])
                    # print("pred : ",pred_buffer[time][id][i])
                    # print("Error_distance : ",distance(gt_buffer[time+i][id],pred_buffer[time][id][i]))
                    time_dist_list.append(
                        distance(gt_buffer[time + i][id], pred_buffer[time][id][i]))
                    count += 1  # count for exist point amount
            ade = sum(time_dist_list) / count
            fde = time_dist_list[-1]
            if id in id_ade_fde.keys():
                id_ade_fde[id].append([ade, fde])
            else:
                id_ade_fde[id] = list()
                id_ade_fde[id].append([ade, fde])

    # write to txt_file or you can add plot pic code
    with open('Ade_fde.txt', 'w+') as f:
        data = str(id_ade_fde)
        f.write(data)
        print('Wrote ADE_FDE in txt file!')


def listener_vehicle():

    global tf_buffer
    rospy.Subscriber(
        '/IPP/Alert',
        DetectedObjectArray,
        record_call_back)

    # spin() simply keeps python from exiting until this node is stopped

    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.spin()


if __name__ == '__main__':
    global pred_buffer, gt_buffer, id_ade_fde
    # pred = prediction, gt = groundtruth
    # pred_buffer = {time: id : [predictions]} (multiple points)
    # gt_buffer = {time : id : [grountruth]} (one point)
    pred_buffer = dict()
    gt_buffer = dict()
    id_ade_fde = dict()
    
    rospy.init_node('calculator')
    tf_map = rospy.get_param(
        '/object_path_prediction/tf_map')
    # IPP input with map-based
    coordinate_type = 'map'
    print("Complete Initialization!")
    try:
        listener_vehicle()
    except rospy.ROSInterruptException:
        pass
    # TODO add time interval of prediction
    atexit.register(calculate)
