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
# from script.ipp_method import create_scene, transform_data
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from environment import Environment, Scene, derivative_of, Node

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def create_scene(scene_ids, present_id):
    obstacle_id_list = []
    max_timesteps = buffer.buffer_frame['frame_id'].max()
    scene = Scene(timesteps=max_timesteps + 1, dt=0.5)
    for node_id in scene_ids:
        node_frequency_multiplier = 1
        node_df = buffer.buffer_frame[buffer.buffer_frame['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            # print 'here!'
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading_ang'].values
        # TODO get obstacle id
        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)

        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)
        if node_df.iloc[0]['type'] == buffer.env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(
                np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            
            
            if np.sum(v_norm, axis = 0)[0]/len(v_norm) < 1.0 and node_id in present_id:
                print('v_norm : ',np.sum(v_norm, axis = 0)[0]/len(v_norm))
                obstacle_id_list.append(int(node_id))
            
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 0.1))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay,
                            ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                            ('heading', 'x'): heading_x,
                            ('heading', 'y'): heading_y,
                            ('heading', 'angle'): heading,
                            ('heading', 'radian'): node_df['heading_rad'].values}
            node_data = pd.DataFrame(
                data_dict, columns=buffer.data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(
                data_dict, columns=buffer.data_columns_pedestrian)

        node = Node(
            node_type=node_df.iloc[0]['type'],
            node_id=node_id,
            data=node_data,
            frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        scene.nodes.append(node)
    buffer.update_prev_id(obstacle_id_list)
    return scene
    
def transform_data(data, tf_map, tf_buffer, rospy):
    
    present_id_list = []
    for obj in data.objects:
        # category = None
        # if obj.classId == 1: #temporary test
        #     category = buffer.env.NodeType.VEHICLE
        #     type_ = "PEDESTRIAN"
        # elif obj.classId == 2 or obj.classId == 3 or obj.classId == 4:
        #     category = buffer.env.NodeType.VEHICLE
        #     type_ = "VEHICLE"
        # else:
        #     continue
        id = int(obj.track.id)
        category = buffer.env.NodeType.VEHICLE
        type_ = "VEHICLE"
        x = obj.center_point.x
        y = obj.center_point.y
        z = obj.center_point.z
        # transform from base_link to map
        if tf_map:
            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_transformed = tf2_geometry_msgs.do_transform_pose(
                pose_stamped, transform)
            x = pose_transformed.pose.position.x
            y = pose_transformed.pose.position.y
            z = pose_transformed.pose.position.z
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
        node_data = pd.Series({'frame_id': buffer.get_curr_frame(),
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
        present_id_list.append(id)
    
    # add temp data
    if short_mem:
        mask_id_list = buffer.add_temp_obstacle(present_id_list)
    else:
        mask_id_list = []
    
    buffer.refresh_buffer()
    # buffer.add_frame_length(len(present_id_list))
    
    return present_id_list,mask_id_list

def predict(data):
    global args,tf_map,tf_buffer
    timer = []

    timer.append(time.time())
    present_id_list, mask_id_list= transform_data(data, tf_map, tf_buffer, rospy)
    # ros parameter
    obj_cnt = len(present_id_list) + len(mask_id_list)
    buffer.add_frame_length(len(present_id_list) + len(mask_id_list))
    scene_ids = map(str, list(present_id_list) + list(mask_id_list))
    present_id_list = map(str,present_id_list)
    # else:
    #     obj_cnt = len(present_id_list)
    #     buffer.add_frame_length(len(present_id_list))
    #     scene_ids = map(str,present_id_list)
    #     present_id_list = scene_ids

    timer.append(time.time())
    
    # previous obstacle id list
    
    scene = create_scene(scene_ids, present_id_list)

    timer.append(time.time())
    scene.calculate_scene_graph(buffer.env.attention_radius,
                                hyperparams['edge_addition_filter'],
                                hyperparams['edge_removal_filter'])
    
    # remove obstacle prediction
    
    timer.append(time.time())
    timesteps = np.array([buffer.get_curr_frame()])

    predictions = eval_stg.predict(scene,
                                   timesteps,
                                   args.get_predict_horizon(),
                                   buffer.env,
                                   num_samples=1,
                                   min_future_timesteps=8,
                                   z_mode=True,
                                   gmm_mode=True,
                                   full_dist=False)
    
    timer.append(time.time())
    buffer.update_frame()
    timer.append(time.time())
    if len(predictions.keys()) < 1:
        return

    t = predictions.keys()[-1]

    for _ , node in enumerate(predictions[t].keys()):
        for obj in data.objects:
            if obj.track.id == int(node.id):
                #  and (obj.track.id == 281 or obj.track.id == 2065)  for debug
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
        queue_size=2)  # /IPP/Alert is TOPIC
    pub.publish(data)

    timer.append(time.time())

    if args.get_print() == 1:
        print ('[RunTime] obj count : ',len(present_id_list))
        print ('[RunTime] Data preprocessing cost time : ',timer[1] - timer[0])
        print ('[RunTime] Create_scene cost time : ',timer[2] - timer[1])
        print ('[RunTime] calculate_scene_graph cost time : ',timer[3] - timer[2])
        print ('[RunTime] Prediction cost time : ',timer[4] - timer[3])
        print ('[RunTime] Update buffer cost time : ',timer[5] - timer[4])
        print ('[RunTime] Pass msg cost time : ',timer[6] - timer[5])
    elif args.get_print() == 2:
        print ('Current time : ', buffer.get_curr_frame())
        print ('Current obj count : ', obj_cnt)
        for _ , node in enumerate(predictions[t].keys()):
            for obj in data.objects:
                if obj.track.id == int(node.id):
                    print ('Current obj id : ',obj.track.id)
                    print ('Current Position : ', (obj.center_point.x,obj.center_point.y))
                    predict_frame = 1
                    for prediction_x_y in predictions[t][node][:][0][0]:
                        predict_frame += 1
                        print ('Prediction ', predict_frame, 'frame : ', prediction_x_y)
                    break
                else:
                    continue
    elif args.get_print() == 3:
        print('Current time : ', buffer.get_curr_frame())
        print('Masked id : ',list(mask_id_list))

def listener_ipp():
    global tf_buffer, tf_listener
    rospy.init_node('object_path_prediction')
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
    global buffer, past_obj, frame_length, args, tf_map, short_mem
    print('Loading model...')
    args = parameter()
    eval_stg, hyperparams = load_model(args.model, ts=args.checkpoint)
    buffer = buffer_data()
    print('Complete Initialization!')

    delay_node = rospy.get_param(
        '/object_path_prediction/delay_node')
    input_source = rospy.get_param(
        '/object_path_prediction/input_topic')
    prediction_horizon = rospy.get_param(
        '/object_path_prediction/prediction_horizon')
    tf_map = rospy.get_param(
        '/object_path_prediction/tf_map')
    show_log = rospy.get_param(
        '/object_path_prediction/print_log')
    short_mem = rospy.get_param(
        '/object_path_prediction/short_mem')
    
    if delay_node == 2:
	    input_topic = '/IPP/delay_Alert'
    elif input_source == 1:
	    input_topic = '/Tracking2D/front_bottom_60'
    elif input_source == 3:
        input_topic = '/Tracking3D'
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
