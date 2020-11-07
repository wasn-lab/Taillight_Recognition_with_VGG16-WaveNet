# -*- coding: UTF-8 -*-
import rospy
from std_msgs.msg import String
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
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

sys.path.insert(0,"./trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline
from environment import Environment, Scene, derivative_of, Node


#tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
#tf_listener = tf2_ros.TransformListener(tf_buffer)
current_frame = 0
count = 0
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

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

# buffer = pd.DataFrame(columns=['frame_id',
#                                  'type',
#                                  'node_id',
#                                  'robot',
#                                  'x', 'y', 'z',
#                                  'length',
#                                  'width',
#                                  'height',
#                                  'heading'])
standardization = {
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
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

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
            scene 隨時間更新的input
            current_time 當前最新的frame_id
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
        self.env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        self.attention_radius = dict()
        self.attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 10.0
        self.attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.VEHICLE)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.PEDESTRIAN)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE)] = 30.0
        self.env.attention_radius = self.attention_radius

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
        self.buffer_frame.sort_values('frame_id', inplace=True)

    def present_all_data(self):
        # print('buffer_length : ', len(self.buffer_frame))
        print self.buffer_frame
        # for i in range(len(self.buffer)):
        #     print('Frame_index : ', i)
        #     print(self.buffer[i])

    def create_scene(self,present_node_id):

        """
            present_node_id : 當前所偵測到的物件id
        """
        #處理資料依照當前的frame所偵測到的node_id 創建 scene_graph
        
        max_timesteps = self.buffer_frame['frame_id'].max()
        # print max_timesteps

        scene = Scene(timesteps=max_timesteps + 1, dt=0.15)
        

        for node_id in present_node_id:
            node_frequency_multiplier = 1
            node_df = self.buffer_frame[self.buffer_frame['node_id'] == node_id]

            # 如果沒有兩個點以上的歷史軌跡則跳過
            if node_df['x'].shape[0] < 2:
                continue
            
            # 全部的frame都要連續
            if not np.all(np.diff(node_df['frame_id']) == 1):
                # print('Occlusion')
                continue  # TODO Make better
            
            node_values = node_df[['x', 'y']].values
            x = node_values[:, 0]
            y = node_values[:, 1]
            heading = node_df['heading_ang'].values
            
            # Kalman filter Agent
            # if node_df.iloc[0]['type'] == env.NodeType.VEHICLE and not node_id == 'ego':
            #     vx = derivative_of(x, self.scene.dt)
            #     vy = derivative_of(y, self.scene.dt)
            #     velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            #     filter_veh = NonlinearKinematicBicycle(dt=self.scene.dt, sMeasurement=1.0)
            #     P_matrix = None
            #     #卡爾曼
            #     for i in range(len(x)):
            #         if i == 0:  # initalize KF
            #             # initial P_matrix
            #             P_matrix = np.identity(4)
            #         elif i < len(x):
            #             # assign new est values
            #             x[i] = x_vec_est_new[0][0]
            #             y[i] = x_vec_est_new[1][0]
            #             heading[i] = x_vec_est_new[2][0]
            #             velocity[i] = x_vec_est_new[3][0]

            #         if i < len(x) - 1:  # no action on last data
            #             # filtering
            #             x_vec_est = np.array([[x[i]],
            #                                 [y[i]],
            #                                 [heading[i]],
            #                                 [velocity[i]]])
            #             z_new = np.array([[x[i + 1]],
            #                             [y[i + 1]],
            #                             [heading[i + 1]],
            #                             [velocity[i + 1]]])
            #             x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
            #                 x_vec_est=x_vec_est,
            #                 u_vec=np.array([[0.], [0.]]),
            #                 P_matrix=P_matrix,
            #                 z_new=z_new
            #             )
            #             P_matrix = P_matrix_new

            #     if pl < 1.0:  # vehicle is "not" moving
            #         x = x[0].repeat(max_timesteps + 1)
            #         y = y[0].repeat(max_timesteps + 1)
            #         heading = heading[0].repeat(max_timesteps + 1)
            
            # TODO rewrite self.scene.dt to real delta time
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            if node_df.iloc[0]['type'] == self.env.NodeType.VEHICLE:
                v = np.stack((vx, vy), axis=-1)
                v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
                heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
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
                            ('heading', '°'): heading,
                            ('heading', 'd°'): node_df['heading_rad'].values}
                node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            else:
                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
                node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
            node.first_timestep = node_df['frame_id'].iloc[0]
            scene.nodes.append(node)
        return scene
            # print(node)

def transform_data(buffer,data):
    # calculate heading part
    global count, past_obj, tf_buffer, tf_listener, current_frame
    present_id_list = []
    count = count + 1
    current_frame = current_frame + 1

    buffer.current_time = current_frame

    for obj in data.objects:
        category = None
        if obj.classId == 1:
            category = buffer.env.NodeType.PEDESTRIAN
        elif obj.classId == 2 or obj.classId == 3 or obj.classId == 4:
            category = buffer.env.NodeType.VEHICLE
        else:
            continue
        x = (obj.bPoint.p0.x + obj.bPoint.p1.x + obj.bPoint.p2.x + obj.bPoint.p3.x + obj.bPoint.p4.x + obj.bPoint.p5.x + obj.bPoint.p6.x + obj.bPoint.p7.x) / 8
        y = (obj.bPoint.p0.y + obj.bPoint.p1.y + obj.bPoint.p2.y + obj.bPoint.p3.y + obj.bPoint.p4.y + obj.bPoint.p5.y + obj.bPoint.p6.y + obj.bPoint.p7.y) / 8
        z = (obj.bPoint.p0.z + obj.bPoint.p1.z + obj.bPoint.p2.z + obj.bPoint.p3.z + obj.bPoint.p4.z + obj.bPoint.p5.z + obj.bPoint.p6.z + obj.bPoint.p7.z) / 8
        # transform from base_link to map
        transform = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
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
        # print node_data
        buffer.update_buffer(node_data)
        present_id_list.append(obj.track.id)

    buffer.refresh_buffer()
    
    return present_id_list

def predict(data):
    global hyperparams,buffer
    # print buffer
    ph = 8
    present_id = transform_data(buffer,data)
    present_id = map(str,present_id)
    scene = buffer.create_scene(present_id)
    # print scene.nodes
    scene.calculate_scene_graph(buffer.env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

    timesteps = np.arange(scene.timesteps)
    # print(buffer.scene)
    predictions = eval_stg.predict(scene,
                                        timesteps,
                                        ph,
                                        buffer.env,
                                        num_samples=1,
                                        min_future_timesteps=8,
                                        z_mode=True,
                                        gmm_mode=True,
                                        full_dist=False)

    print predictions
# def data_structure():

def listener_ipp():
    global tf_buffer, tf_listener
    rospy.init_node('ipp_transform_data')
    rospy.Subscriber('/Tracking2D/front_bottom_60', DetectedObjectArray, predict)
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

    print 'Loading model...'
    args = parameter()
    eval_stg, hyperparams = load_model(args.model, ts=args.checkpoint)
    buffer = buffer_data()
    print 'Complete loading model!'

    # for i in range(10):
    #     heading = math.atan(i*(-1))
    #     print(i,heading, math.degrees(heading))
    # info = "frame_id" + "," + "type" + "," + "node_id" + "," + "robot" + "," + "x" + "," + "y" + "," + "z" + "," + "length" + "," + "width" + "," + "height" + "," + "heading"
    # print info
    try:
        listener_ipp()
    except rospy.ROSInterruptException:
        pass
