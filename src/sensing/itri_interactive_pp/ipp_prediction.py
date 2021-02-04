#! ./sandbox/bin/python2.7
# coding=utf-8
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import time

sys.path.insert(0, "./trajectron")

import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import PoseStamped
from tf.transformations import euler_from_quaternion

from msgs.msg import PointXY
from msgs.msg import PathPrediction
from msgs.msg import DetectedObject
from msgs.msg import DetectedObjectArray
from std_msgs.msg import String
import rospy
from tqdm import tqdm
import math

from script.ipp_class import parameter, buffer_data
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from environment import Scene, Node

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def transform_data(buffer, data):
    present_id_list = []

    for obj in data.objects:
        category = None
        if obj.classId == 1: #temporary test
            category = buffer.env.NodeType.VEHICLE
            type_ = "VEHICLE"
        elif obj.classId == 2 or obj.classId == 3 or obj.classId == 4:
            category = buffer.env.NodeType.VEHICLE
            type_ = "VEHICLE"
        else:
            continue
        x = obj.center_point.x
        y = obj.center_point.y
        z = obj.center_point.z

        # transform from base_link to map
        # if transformer:
        #     transform = tf_buffer.lookup_transform(
        #         'map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
        #     pose_stamped = PoseStamped()
        #     pose_stamped.pose.position.x = x
        #     pose_stamped.pose.position.y = y
        #     pose_stamped.pose.position.z = z
        #     pose_transformed = tf2_geometry_msgs.do_transform_pose(
        #         pose_stamped, transform)
        #     x = pose_transformed.pose.position.x
        #     y = pose_transformed.pose.position.y
        #     z = pose_transformed.pose.position.z
        # print 'x: ', x
        # print(pose_transformed.pose.position.x, pose_transformed.pose.position.y, pose_transformed.pose.position.z)

        length = math.sqrt(
            math.pow(
                (obj.bPoint.p4.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p4.y -
                 obj.bPoint.p0.y),
                2))
        width = math.sqrt(
            math.pow(
                (obj.bPoint.p3.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p3.y -
                 obj.bPoint.p0.y),
                2))
        height = math.sqrt(
            math.pow(
                (obj.bPoint.p1.x -
                 obj.bPoint.p0.x),
                2) +
            math.pow(
                (obj.bPoint.p1.y -
                 obj.bPoint.p0.y),
                2))
        diff_x = 0
        diff_y = 0
        heading = 0
        # for CH object.yaw
        heading = obj.distance
        heading_rad = math.radians(heading)
        # heading_rad = 0
        # for old_obj in past_obj:
        #     sp = old_obj.split(",")
        #     if obj.track.id == int(sp[2]):
        #         diff_x = x - float(sp[4])
        #         diff_y = y - float(sp[5])
        #         if diff_x == 0:
        #             heading = 90
        #         else:
        #             heading = abs(math.degrees(math.atan(diff_y / diff_x)))
        #         # print(diff_x,diff_y,diff_y/diff_x,heading)
        #         if diff_x == 0 and diff_y == 0:
        #             heading = 0
        #         elif diff_x >= 0 and diff_y >= 0:
        #             heading = heading
        #         elif diff_x >= 0 and diff_y < 0:
        #             heading = 360 - heading
        #         elif diff_x < 0 and diff_y >= 0:
        #             heading = 180 - heading
        #         else:
        #             heading = 180 + heading
        #         if heading > 180:
        #             heading = heading - 360
        #         heading_rad = math.radians(heading)
        # info = str(buffer.get_buffer_frame()) + "," + type_ + "," + str(obj.track.id) + "," + "False" + "," + str(x) + \
        #     "," + str(y) + "," + str(z) + "," + str(length) + "," + str(width) + "," + str(height) + "," + str(heading)
        # past_obj.append(info)
        # print 'ros method heading : ',yaw
        # print 'our method heading : ',heading
        node_data = pd.Series({'frame_id': buffer.get_buffer_frame(),
                               'type': category,
                               'node_id': str(obj.track.id),
                               'robot': False,  # frame_data.loc[i]['robot']
                               'x': x,
                               'y': y,
                               'z': z,
                               'length': length,
                               'width': width,
                               'height': height,
                               'heading_ang': heading,
                               'heading_rad': heading_rad})
        
        buffer.update_buffer(node_data)
        present_id_list.append(obj.track.id)
    # print(present_id_list)
    buffer.refresh_buffer()
    buffer.add_frame_length(len(present_id_list))
    return present_id_list


def predict(data):
    global args
    timer = []

    prev = time.time()
    present_id = transform_data(buffer, data)
    timer.append(time.time() - prev)

    present_id = map(str, present_id)
    scene = buffer.create_scene(present_id)

    timer.append(time.time() - timer[-1])
    scene.calculate_scene_graph(buffer.env.attention_radius,
                                hyperparams['edge_addition_filter'],
                                hyperparams['edge_removal_filter'])
    timer.append(time.time() - timer[-1])
    timesteps = np.array([buffer.get_buffer_frame()])

    predictions = eval_stg.predict(scene,
                                   timesteps,
                                   args.get_predict_horizon(),
                                   buffer.env,
                                   num_samples=1,
                                   min_future_timesteps=8,
                                   z_mode=True,
                                   gmm_mode=True,
                                   full_dist=False)
    
    timer.append(time.time() - timer[-1])
    buffer.update_frame()
    timer.append(time.time() - timer[-1])
    if len(predictions.keys()) < 1:
        return

    t = predictions.keys()[-1]

    for _ , node in enumerate(predictions[t].keys()):
        for obj in data.objects:
            obj.track.forecasts = []
            if obj.track.id == int(node.id):
                for prediction_x_y in predictions[t][node][:][0][0]:

                    forecasts_item = PathPrediction()
                    forecasts_item.position.x = np.float32(prediction_x_y[0])
                    forecasts_item.position.y = np.float32(prediction_x_y[1])

                    obj.track.forecasts.append(forecasts_item)
                    obj.track.is_ready_prediction = True
                    # print(prediction_x_y)
                break
            else:
                continue

    pub = rospy.Publisher(
        '/IPP/Alert',
        DetectedObjectArray,
        queue_size=1)  # /IPP/Alert is TOPIC
    pub.publish(data)

    timer.append(time.time() - timer[-1])

    if args.get_print() == 1:
        print '[RunTime] obj count : ',len(present_id)
        print '[RunTime] Data preprocessing cost time : ',timer[0]
        print '[RunTime] Create_scene cost time : ',timer[1]
        print '[RunTime] calculate_scene_graph cost time : ',timer[2]
        print '[RunTime] Prediction cost time : ',timer[3]
        print '[RunTime] Update buffer cost time : ',timer[4]
        print '[RunTime] Pass msg cost time : ',timer[5]
    elif args.get_print() == 2:
        print 'Current time : ', buffer.get_buffer_frame()
        for _ , node in enumerate(predictions[t].keys()):
            for obj in data.objects:
                if obj.track.id == int(node.id):
                    print 'Current Position : ', (obj.center_point.x,obj.center_point.y)
                    predict_frame = 1
                    for prediction_x_y in predictions[t][node][:][0][0]:

                        forecasts_item = PathPrediction()
                        forecasts_item.position.x = np.float32(prediction_x_y[0])
                        forecasts_item.position.y = np.float32(prediction_x_y[1])

                        obj.track.forecasts.append(forecasts_item)
                        obj.track.is_ready_prediction = True
                        print 'Prediction ', predict_frame, 'frame : ', prediction_x_y
                        predict_frame += 1
                    break
                else:
                    continue

def listener_ipp():
    global tf_buffer, tf_listener
    rospy.init_node('ipp_transform_data')
    rospy.Subscriber(args.get_source(), DetectedObjectArray, predict)
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def load_model(model_dir, ts=100):
    global hyperparams
    model_registrar = ModelRegistrar(model_dir, 'cuda:1')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cuda:1')

    trajectron.set_environment()
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == '__main__':
    ''' =====================
          loading_model_part
          frame_length for refreshing buffer
        ===================== '''
    global buffer, past_obj, frame_length, args
    print('Loading model...')
    args = parameter()
    eval_stg, hyperparams = load_model(args.model, ts=args.checkpoint)
    buffer = buffer_data()
    print('Complete loading model!')

    delay_node = rospy.get_param(
        '/object_path_prediction/delay_node')
    input_source = rospy.get_param(
        '/object_path_prediction/input_topic')
    prediction_horizon = rospy.get_param(
        '/object_path_prediction/prediction_horizon')
    show_log = rospy.get_param(
        '/object_path_prediction/print_log')

    if delay_node == 2:
	    input_topic = '/IPP/delay_Alert'
    elif input_source == 1:
	    input_topic = '/Tracking2D/front_bottom_60'
    else:
	    input_topic = '/PathPredictionOutput'
    
    args.set_params(input_topic, prediction_horizon, show_log)
    past_obj = []
    frame_length = []

    # for i in range(10):
    #     heading = math.atan(i*(-1))
    #     print(i,heading, math.degrees(heading))
    # info = "frame_id" + "," + "type" + "," + "node_id" + "," + "robot" + "," + "x" + "," + "y" + "," + "z" + "," + "length" + "," + "width" + "," + "height" + "," + "heading"
    # print info
    try:
        listener_ipp()
    except rospy.ROSInterruptException:
        pass
