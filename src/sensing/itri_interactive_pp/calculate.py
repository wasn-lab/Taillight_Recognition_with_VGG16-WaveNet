#! ./sandbox/bin/python2.7
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import time
import numpy as np
import pandas as pd
import math

def record_call_back(data):
    id_pred = dict()
    id_gt = dict()
    pred = list()
    for obj in data.objects:
        id = int(obj.track.id)
        x_gt = obj.center_point.x
        y_gt = obj.center_point.y
        id_gt[id] = [x_gt,y_gt]
        if obj.track.is_ready_prediction:
            for track_point in obj.track.forecasts:
                x_pred = track_point.position.x
                y_pred = track_point.position.y
                pred.append([x_pred,y_pred])
        id_pred[id] = pred
    pred_buffer[timer] = id_pred
    gt_buffer[timer] = id_gt
    timer+=1

def calculate(pred_horizon):
    # calculate part
    for time in range(timer):
        time_dist_list = list()
        for id in pred_buffer[time].keys() :
            count = 0
            for i in range(pred_horizon) :
                # check for the value is exist or not (If id switch in future the id won't exist)
                if id in gt_buffer[time+i].keys() : 
                    time_dist_list.append(math.dist(gt_buffer[time+i][id],pred_buffer[time][id][i]))
                    count += 1  # count for exist point amount
            ade = sum(time_dist_list)/count
            fde = time_dist_list[-1]
            if id in id_ade_fde.keys():
                id_ade_fde[id].append(ade,fde)
            else :
                id_ade_fde[id] = list()
                id_ade_fde[id].append(ade,fde)
    
    # write to txt_file or you can add plot pic code
    with open('Ade_fde.txt','w+') as f :
        data = str(id_ade_fde)
        f.write(data)
    


def listener_vehicle():
    rospy.Subscriber(
        '/IPP/Alert',
        DetectedObjectArray,
        record_call_back)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    global input_source,coordinate_type,timer,buffer
    # pred = prediction, gt = groundtruth
    pred_buffer = dict()
    gt_buffer = dict()
    id_ade_fde = dict()
    # init time counter
    timer = 0
    rospy.init_node('Ipp_Marker')
    # IPP input with map-based
    coordinate_type = 'map'
        
    try:
        listener_vehicle()
    except rospy.ROSInterruptException:
        pass
    # TODO add time interval of prediction
    atexit.register(calculate)


## TODO 
# - Create two callback function and add ros param fps to know the time interval
# - No publisher
# - 自製cuurent time 計算 (考慮掉幀的問題) message 中有dt
# - 用buffer 紀錄 預測位置 和 當前座標位置 程式結束時計算結果
    
