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
                                                  'timer_sec',
                                                  'type',
                                                  'node_id',
                                                  'robot',
                                                  'x', 'y', 'z',
                                                  'length',
                                                  'width',
                                                  'height',
                                                  'heading_ang',
                                                  'heading_rad'])
        self.buffer_predicted_frame = None
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
        self.output_data_column = pd.MultiIndex.from_product([['frame', 'node'],['id']])
        self.output_data_column = self.output_data_column.append(self.data_columns_vehicle.copy())
        
        self.current_frame = 0
        self.frame_length = []
        self.previous_id_list = []
        # dict keys = id, values = keep_time 
        self.obstacle_buffer = dict()
        # dict keys = id, values = (x,y) -> tuple
        self.heading_buffer = dict()

    def add_temp_obstacle(self,present_id):
        '''=================
        Get node_id's last frame and duplicate it replace frame_num
        
        present_id -> list: current obstacle id 
        
        ===================='''
        # print('previous_id_list : ',self.previous_id_list)
        # print('present_id : ',present_id)

        # get mask obstacle object
        mask_id = set(self.previous_id_list) - set(present_id)
        for node_id in mask_id:
            if node_id not in self.obstacle_buffer.keys():
                self.obstacle_buffer[node_id] = 1
                
            node_df = self.buffer_frame[self.buffer_frame['node_id'] == node_id]
            node_df = node_df[node_df['frame_id'] ==  self.get_attribute("current_frame") - 1].copy()
            node_df = node_df.replace('frame_id',self.get_attribute("current_frame"))
            # print('Node_id : ',node_id,'copy data : ',node_df)
            self.update_buffer(node_df)

        for node_id in self.obstacle_buffer.keys():
            if node_id in present_id:
                continue
            if self.obstacle_buffer[node_id] <= 3 and self.obstacle_buffer[node_id] >=1:
                print('mask_id : ',node_id)
                self.obstacle_buffer[node_id] += 1
            else:
                del self.obstacle_buffer[node_id]
                continue
            node_df = self.buffer_frame[self.buffer_frame['node_id'] == node_id]
            node_df = node_df[node_df['frame_id'] ==  self.get_attribute("current_frame") - 1].copy()
            node_df = node_df.replace('frame_id',self.get_attribute("current_frame"))
            # print('Node_id : ',node_id,'copy data : ',node_df)
            self.update_buffer(node_df)
        
        return list(mask_id)

    def refresh_buffer(self):
        # If frame_id < current_time - 11 remove the data
        self.buffer_frame = self.buffer_frame.reset_index(drop = True)
        if len(self.frame_length) > 11:
            if self.frame_length[0] > 0:
                self.buffer_frame = self.buffer_frame.drop(range(0,self.frame_length[0]))
            del self.frame_length[0]
            
    def add_frame_length(self,frame_obj_nb):
        self.frame_length.append(frame_obj_nb)
    
    def get_attribute(self,attribute_name):
        for attribute in self.__dict__.keys():
            if attribute == attribute_name:
                value = getattr(self,attribute)
                return value
        print("attribute_name : ", attribute_name, " not found !")
    
    def update_heading_buffer(self,datas):
        self.heading_buffer.clear()
        for data in datas:
            self.heading_buffer[data[0]] = (data[1],data[2])
        
    def update_prev_id(self,prev_id_list):
        self.previous_id_list = list(prev_id_list)
    
    def update_frame(self):
        self.current_frame = self.current_frame + 1

    def update_buffer(self, data):
        '''
            data : pd.series
        '''
        # update T frame objects in last element array
        self.buffer_frame = self.buffer_frame.append(data, ignore_index=True)
        
    def update_predict_frame(self, data):
        '''
            data : pd.series
        '''
        if self.buffer_predicted_frame is not None:
            self.buffer_predicted_frame = pd.concat([self.buffer_predicted_frame,data])
        else:
            self.buffer_predicted_frame = data

    def reset_buffers(self):
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
        self.current_frame = 0
        self.frame_length = []
        self.previous_id_list = []
        self.obstacle_buffer = dict()
        
    # def print_buffer(self):
    #     print(self.buffer_frame)


