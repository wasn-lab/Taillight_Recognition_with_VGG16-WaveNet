# coding=utf-8
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from msgs.msg import PathPrediction
from msgs.msg import PointXY
import math
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import PoseStamped

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import time

sys.path.insert(0,"./trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from environment import Environment, Scene, derivative_of, Node

current_frame = 0
past_obj = []
tf_buffer = None
tf_listener = None
hyperparams = None
buffer = None

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class parameter():
    def __init__(self):
        self.model = 'models/int_ee'
        self.checkpoint = 12
        self.data = '../processed/nuScenes_test_full.pkl'
        self.output_path = 'results'
        self.output_tag = 'int_ee'
        self.node_type = 'VEHICLE'
        self.prediction_horizon = 6

class buffer_data():
    def __init__(self):
        
        '''

        '''

        self.buffer_frame = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading_ang',
                                 'heading_rad'])
        # self.scene = None
        self.current_time = None
        standardization  = {
            'PEDESTRIAN': {
                'position': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                },
                'velocity': {
                    'x': {'mean': 0, 'std': 2},
                    'y': {'mean': 0, 'std': 2}
                },
                'acceleration': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1}
                }
            },
            'VEHICLE': {
                'position': {
                    'x': {'mean': 0, 'std': 80},
                    'y': {'mean': 0, 'std': 80}
                },
                'velocity': {
                    'x': {'mean': 0, 'std': 15},
                    'y': {'mean': 0, 'std': 15},
                    'norm': {'mean': 0, 'std': 15}
                },
                'acceleration': {
                    'x': {'mean': 0, 'std': 4},
                    'y': {'mean': 0, 'std': 4},
                    'norm': {'mean': 0, 'std': 4}
                },
                'heading': {
                    'x': {'mean': 0, 'std': 1},
                    'y': {'mean': 0, 'std': 1},
                    'angle': {'mean': 0, 'std': np.pi},
                    'radian': {'mean': 0, 'std': 1}
                }
            }
        }
        self.env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        self.attention_radius = dict()
        self.attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 10.0
        self.attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.VEHICLE)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.PEDESTRIAN)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE)] = 30.0
        self.env.attention_radius = self.attention_radius
        self.data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
        self.data_columns_vehicle = self.data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading','angle'), ('heading', 'radian')]))
        self.data_columns_vehicle = self.data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
        self.data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    def update_buffer(self,data):
        
        '''
            data : pd.series
        '''
        # print(self.current_time)
        # update T frame objects in last element array
        self.buffer_frame = self.buffer_frame.append(data, ignore_index=True)
        # self.present_all_data()
            
    def refresh_buffer(self):
        # If frame_id < current_time - 11 remove the data
        last_time = self.current_time - 11
        self.buffer_frame = self.buffer_frame[self.buffer_frame['frame_id'] >= last_time]
        # print(self.buffer_frame)
        self.buffer_frame.sort_values('frame_id', inplace=True)

    def present_all_data(self):
        print('buffer_length : ', len(self.buffer_frame))
        # print self.buffer_frame[self.buffer_frame['node_id']=='1754']
        # for i in range(len(self.buffer)):
        #     print('Frame_index : ', i)
        #     print(self.buffer[i])

    def create_scene(self,present_node_id):
        max_timesteps = self.buffer_frame['frame_id'].max()
        # print max_timesteps
        scene = Scene(timesteps=max_timesteps + 1, dt=0.5)
        for node_id in pd.unique(self.buffer_frame['node_id']):
            node_frequency_multiplier = 1
            node_df = self.buffer_frame[self.buffer_frame['node_id'] == node_id]

            if node_df['x'].shape[0] < 2:
                continue
            
            if not np.all(np.diff(node_df['frame_id']) == 1):
                # print('Occlusion')
                continue  # TODO Make better
            
            node_values = node_df[['x', 'y']].values
            x = node_values[:, 0]
            y = node_values[:, 1]
            heading = node_df['heading_ang'].values
            # TODO rewrite self.scene.dt to real delta time
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)
            # print("Dt: ",scene.dt,"Vx : ", vx,"Vy : ", vy,"Ax : ", ax,"Ax : ", ay)  
            if node_df.iloc[0]['type'] == self.env.NodeType.VEHICLE:
                v = np.stack((vx, vy), axis=-1)
                v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis= -1, keepdims=True)
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
                node_data = pd.DataFrame(data_dict, columns=self.data_columns_vehicle)
            else:
                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
                node_data = pd.DataFrame(data_dict, columns=self.data_columns_pedestrian)

            node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
            node.first_timestep = node_df['frame_id'].iloc[0]
            scene.nodes.append(node)

        return scene

def transform_data(buffer,data):
    # calculate heading part
    global past_obj, tf_buffer, tf_listener, current_frame
    present_id_list = []
    

    buffer.current_time = current_frame

    for obj in data.objects:
        # if obj.track.id != 1754:
        #     continue
        category = None
        if obj.classId == 1:
            category = buffer.env.NodeType.PEDESTRIAN
            type_ = "PEDESTRIAN"
        elif obj.classId == 2 or obj.classId == 3 or obj.classId == 4:
            category = buffer.env.NodeType.VEHICLE
            type_ = "VEHICLE"
        else:
            continue
        x = (obj.bPoint.p0.x + obj.bPoint.p1.x + obj.bPoint.p2.x + obj.bPoint.p3.x + obj.bPoint.p4.x + obj.bPoint.p5.x + obj.bPoint.p6.x + obj.bPoint.p7.x) / 8
        y = (obj.bPoint.p0.y + obj.bPoint.p1.y + obj.bPoint.p2.y + obj.bPoint.p3.y + obj.bPoint.p4.y + obj.bPoint.p5.y + obj.bPoint.p6.y + obj.bPoint.p7.y) / 8
        z = (obj.bPoint.p0.z + obj.bPoint.p1.z + obj.bPoint.p2.z + obj.bPoint.p3.z + obj.bPoint.p4.z + obj.bPoint.p5.z + obj.bPoint.p6.z + obj.bPoint.p7.z) / 8
        # transform from base_link to map
        # transform = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
        # pose_stamped = PoseStamped()
        # pose_stamped.pose.position.x = x
        # pose_stamped.pose.position.y = y
        # pose_stamped.pose.position.z = z
        # pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        # print(pose_transformed.pose.position.x, pose_transformed.pose.position.y, pose_transformed.pose.position.z)
        length = math.sqrt(math.pow((obj.bPoint.p4.x - obj.bPoint.p0.x), 2) + math.pow((obj.bPoint.p4.y - obj.bPoint.p0.y), 2))
        width = math.sqrt(math.pow((obj.bPoint.p3.x - obj.bPoint.p0.x), 2) + math.pow((obj.bPoint.p3.y - obj.bPoint.p0.y), 2))
        height = math.sqrt(math.pow((obj.bPoint.p1.x - obj.bPoint.p0.x), 2) + math.pow((obj.bPoint.p1.y - obj.bPoint.p0.y), 2))
        # x = pose_transformed.pose.position.x
        # y = pose_transformed.pose.position.y
        # z = pose_transformed.pose.position.z
        diff_x = 0
        diff_y = 0
        heading = 0
        heading_rad = 0
        for old_obj in past_obj:
            sp = old_obj.split(",")
            if obj.track.id == int(sp[2]):
                diff_x = x - float(sp[4])
                diff_y = y - float(sp[5])
                if diff_x == 0:
                    heading = 90
                else:
                    heading = abs(math.degrees(math.atan(diff_y/diff_x)))
                # print(diff_x,diff_y,diff_y/diff_x,heading)
                if diff_x == 0 and diff_y == 0:
                    heading = 0
                elif diff_x >= 0 and diff_y >= 0:
                    heading = heading
                elif diff_x >= 0 and diff_y < 0:
                    heading = 360 - heading
                elif diff_x < 0 and diff_y >= 0:
                    heading = 180 - heading
                else:
                    heading = 180 + heading
                if heading > 180:
                    heading = heading - 360
                heading_rad = math.radians(heading)
        info = str(current_frame) + "," + type_ + "," + str(obj.track.id) + "," + "False" + "," + str(x) + "," + str(y) + "," + str(z) + "," + str(length) + "," + str(width) + "," + str(height) + "," + str(heading)
        past_obj.append(info)
        # if diff_x != 0:
        #    print(diff_x,diff_y,diff_y/diff_x,heading)

        node_data = pd.Series({'frame_id' : current_frame,
                        'type': category,
                        'node_id': str(obj.track.id),
                        'robot': False, #frame_data.loc[i]['robot']
                        'x': x,
                        'y': y,
                        'z': z,
                        'length': length,
                        'width': width,
                        'height': height,
                        'heading_ang': heading,
                        'heading_rad': heading_rad})
        # if obj.track.id==1754:
        #     print node_data
        buffer.update_buffer(node_data)
        present_id_list.append(obj.track.id)
    current_frame = current_frame + 1
    buffer.refresh_buffer()
    
    return present_id_list

def predict(data):
    global hyperparams,buffer
    # print buffer
    ph = 2
    present_id = transform_data(buffer,data)
    present_id = map(str,present_id)
    scene = buffer.create_scene(present_id)
    
    scene.calculate_scene_graph(buffer.env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
    
    timesteps = np.array([buffer.current_time])
    # print timesteps
    # print buffer.current_time
    print('====')
    print('current_time : ',buffer.current_time)
    
    # for node in scene.nodes:
    #     if node.id == '1754':
    #         print "node_id: ",node.id
    #         print "node_data_x: ",node.data.data[-1:,0]
    #         print "node_data_y: ",node.data.data[-1:,1]
            # print "node_data_vx: ",node.data.data[-1:,2]
            # print "node_data_vy: ",node.data.data[-1:,3]
            # print "node_data_ax: ",node.data.data[-1:,4]
            # print "node_data_ay: ",node.data.data[-1:,5]
            # print "node_data_heading_angle: ",node.data.data[-1:,8]
            # print "node_data_heading_radian: ",node.data.data[-1:,9]
            
    predictions = eval_stg.predict(scene,
                                        timesteps,
                                        ph,
                                        buffer.env,
                                        num_samples=1,
                                        min_future_timesteps=8,
                                        z_mode=True,
                                        gmm_mode=True,
                                        full_dist=False)
                                        
    if len(predictions.keys())<1:
        return
    t = predictions.keys()[-1]

    # for t in predictions.keys():
    #     for node in predictions[t].keys():
    #         if node.id == '1754':
    #             print('Predict: ', predictions[t][node][:][0][0])

    # print '===='
    for index, node in enumerate(predictions[t].keys()):
        for obj in data.objects:
            if obj.track.id == int(node.id):
                print('object id', obj.track.id)
                print('node id', node.id)
                for prediction_x_y in predictions[t][node][:][0][0]:

                    forecasts_item = PathPrediction()
                    forecasts_item.position.x = np.float32(prediction_x_y[0])
                    forecasts_item.position.y = np.float32(prediction_x_y[1])

                    obj.track.forecasts.append(forecasts_item)
                    obj.track.is_ready_prediction = True
                break
            else:
                continue

    pub = rospy.Publisher('/IPP/Alert', DetectedObjectArray, queue_size=1) # /IPP/Alert is TOPIC
    pub.publish(data)

    print('====')

def listener_ipp():
    global tf_buffer, tf_listener
    rospy.init_node('ipp_transform_data')
    rospy.Subscriber('/IPP/delay_Alert', DetectedObjectArray, predict)
    tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def load_model(model_dir, ts=100):
    global hyperparams
    model_registrar = ModelRegistrar(model_dir, 'cuda:0')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cuda:0')

    trajectron.set_environment()
    trajectron.set_annealing_params()
    return trajectron, hyperparams

if __name__ == '__main__':
    # global buffer
    #global tf_buffer, tf_listener
    #tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
    #tf_listener = tf2_ros.TransformListener(tf_buffer)

    ''' =====================
          loading_model_part
        ===================== '''

    print('Loading model...')
    args = parameter()
    eval_stg, hyperparams = load_model(args.model, ts=args.checkpoint)
    buffer = buffer_data()
    print('Complete loading model!')

    # for i in range(10):
    #     heading = math.atan(i*(-1))
    #     print(i,heading, math.degrees(heading))
    # info = "frame_id" + "," + "type" + "," + "node_id" + "," + "robot" + "," + "x" + "," + "y" + "," + "z" + "," + "length" + "," + "width" + "," + "height" + "," + "heading"
    # print info
    try:
        listener_ipp()
    except rospy.ROSInterruptException:
        pass
