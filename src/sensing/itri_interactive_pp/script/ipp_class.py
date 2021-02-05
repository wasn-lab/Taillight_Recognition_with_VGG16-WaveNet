import pandas as pd
import numpy as np
from environment import Environment, Scene, derivative_of, Node

class parameter():
    def __init__(self):
        self.model = 'models/int_ee'
        self.checkpoint = 12
        self.node_type = 'VEHICLE'
        self.log_printer = None
        self.input_topic = None
        self.transformer = None
        self.prediction_horizon = None

    def set_params(self,input, prediction_horizon, print_log):
        self.input_topic = input
        self.prediction_horizon = prediction_horizon
        self.log_printer = print_log

    def get_print(self):
        return self.log_printer
    
    def get_source(self):
        return self.input_topic
    
    def get_predict_horizon(self):
        return self.prediction_horizon

class buffer_data():
    def __init__(self):
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
                    'angle': {'mean': 0, 'std': np.pi},
                    'radian': {'mean': 0, 'std': 1}
                }
            }
        }
        self.env = Environment(
            node_type_list=[
                'VEHICLE',
                'PEDESTRIAN'],
            standardization=standardization)
        self.attention_radius = dict()
        self.attention_radius[(self.env.NodeType.PEDESTRIAN,
                               self.env.NodeType.PEDESTRIAN)] = 10.0
        self.attention_radius[(self.env.NodeType.PEDESTRIAN,
                               self.env.NodeType.VEHICLE)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE,
                               self.env.NodeType.PEDESTRIAN)] = 20.0
        self.attention_radius[(self.env.NodeType.VEHICLE,
                               self.env.NodeType.VEHICLE)] = 30.0
        self.env.attention_radius = self.attention_radius
        self.data_columns_vehicle = pd.MultiIndex.from_product(
            [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
        self.data_columns_vehicle = self.data_columns_vehicle.append(
            pd.MultiIndex.from_tuples([('heading', 'angle'), ('heading', 'radian')]))
        self.data_columns_vehicle = self.data_columns_vehicle.append(
            pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
        self.data_columns_pedestrian = pd.MultiIndex.from_product(
            [['position', 'velocity', 'acceleration'], ['x', 'y']])
        
        self.current_frame = 0
        self.frame_length = []

    def update_frame(self):
        self.current_frame = self.current_frame + 1

    def get_buffer_frame(self):
        return self.current_frame

    def update_buffer(self, data):
        '''
            data : pd.series
        '''
        # update T frame objects in last element array
        self.buffer_frame = self.buffer_frame.append(data, ignore_index=True)

    def refresh_buffer(self):
        # If frame_id < current_time - 11 remove the data
        if len(self.frame_length) > 11:
            tmp = self.buffer_frame.copy()
            tmp.drop(range(0,self.frame_length[0]))
            del self.frame_length[0]
            
    def add_frame_length(self,frame_obj_nb):
        self.frame_length.append(frame_obj_nb)

    def create_scene(self, present_node_id):
        max_timesteps = self.buffer_frame['frame_id'].max()
        scene = Scene(timesteps=max_timesteps + 1, dt=0.5)
        for node_id in present_node_id:
            node_frequency_multiplier = 1
            node_df = self.buffer_frame[self.buffer_frame['node_id'] == node_id]

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
            # TODO rewrite self.scene.dt to real delta time
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)
            if node_df.iloc[0]['type'] == self.env.NodeType.VEHICLE:
                v = np.stack((vx, vy), axis=-1)
                v_norm = np.linalg.norm(
                    np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
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
                    data_dict, columns=self.data_columns_vehicle)
            else:
                data_dict = {('position', 'x'): x,
                             ('position', 'y'): y,
                             ('velocity', 'x'): vx,
                             ('velocity', 'y'): vy,
                             ('acceleration', 'x'): ax,
                             ('acceleration', 'y'): ay}
                node_data = pd.DataFrame(
                    data_dict, columns=self.data_columns_pedestrian)

            node = Node(
                node_type=node_df.iloc[0]['type'],
                node_id=node_id,
                data=node_data,
                frequency_multiplier=node_frequency_multiplier)
            node.first_timestep = node_df['frame_id'].iloc[0]
            scene.nodes.append(node)

        return scene