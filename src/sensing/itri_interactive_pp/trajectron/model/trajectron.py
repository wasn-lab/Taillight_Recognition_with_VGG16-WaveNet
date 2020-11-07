# -*- coding: UTF-8 -*-

import torch
import numpy as np
from mgcvae import MultimodalGenerativeCVAE
from dataset import get_timesteps_data, restore
from environment.node_type import NodeType 


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = {'batch_size': 1, 'grad_clip': 1.0, 'learning_rate_style': 'exp', 'learning_rate': 0.003, 'min_learning_rate': 1e-05, 'learning_decay_rate': 0.9999, 'prediction_horizon': 6, 'minimum_history_length': 1, 'maximum_history_length': 8, 'map_encoder': {'VEHICLE': {'heading_state_index': 6, 'patch_size': [50, 10, 50, 90], 'map_channels': 3, 'hidden_channels': [10, 20, 10, 1], 'output_size': 32, 'masks': [5, 5, 5, 3], 'strides': [2, 2, 1, 1], 'dropout': 0.5}}, 'k': 1, 'k_eval': 25, 'kl_min': 0.07, 'kl_weight': 100.0, 'kl_weight_start': 0, 'kl_decay_rate': 0.99995, 'kl_crossover': 400, 'kl_sigmoid_divisor': 4, 'rnn_kwargs': {'dropout_keep_prob': 0.75}, 'MLP_dropout_keep_prob': 0.9, 'enc_rnn_dim_edge': 32, 'enc_rnn_dim_edge_influence': 32, 'enc_rnn_dim_history': 32, 'enc_rnn_dim_future': 32, 'dec_rnn_dim': 128, 'q_z_xy_MLP_dims': None, 'p_z_x_MLP_dims': 32, 'GMM_components': 1, 'log_p_yt_xz_max': 6, 'N': 1, 'K': 25, 'tau_init': 2.0, 'tau_final': 0.05, 'tau_decay_rate': 0.997, 'use_z_logit_clipping': True, 'z_logit_clip_start': 0.05, 'z_logit_clip_final': 5.0, 'z_logit_clip_crossover': 300, 'z_logit_clip_divisor': 5, 'dynamic': {'PEDESTRIAN': {'name': 'SingleIntegrator', 'distribution': True, 'limits': {}}, 'VEHICLE': {'name': 'Unicycle', 'distribution': True, 'limits': {'max_a': 4, 'min_a': -5, 'max_heading_change': 0.7, 'min_heading_change': -0.7}}}, 'state': {'PEDESTRIAN': {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y']}, 'VEHICLE': {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y'], 'heading': ['°', 'd°']}}, 'pred_state': {'VEHICLE': {'position': ['x', 'y']}, 'PEDESTRIAN': {'position': ['x', 'y']}}, 'log_histograms': False, 'dynamic_edges': 'yes', 'edge_state_combine_method': 'sum', 'edge_influence_combine_method': 'attention', 'edge_addition_filter': [0.25, 0.5, 0.75, 1.0], 'edge_removal_filter': [1.0, 0.0], 'offline_scene_graph': 'yes', 'incl_robot_node': False, 'node_freq_mult_train': True, 'node_freq_mult_eval': False, 'scene_freq_mult_train': False, 'scene_freq_mult_eval': False, 'scene_freq_mult_viz': False, 'edge_encoding': True, 'use_map_encoding': False, 'augment': True, 'override_attention_radius': [], 'map_enc_dropout': 0.0}
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.node_type_dict = ['VEHICLE','PEDESTRIAN']
        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph =  self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self):
        # self.env = env

        self.node_models_dict.clear()
        # edge_types = env.get_edge_types()
        # print(edge_types)
        VEHICLE = NodeType('VEHICLE',0)
        PEDESTRIAN = NodeType('PEDESTRIAN',0)
        edge_types = [(VEHICLE, VEHICLE), (VEHICLE, PEDESTRIAN), (PEDESTRIAN, VEHICLE), (PEDESTRIAN, PEDESTRIAN)]
        for node_type in self.node_type_dict:
            ### edit by onebone
            # print('node type')
            # print(type(node_type))
            # print(node_type)
            
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def predict(self,
                scene,
                timesteps,
                ph,
                env,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.node_type_dict:
            if node_type not in self.pred_state:
                continue
            
            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch
            
            
            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)
            # print('current_time_stamp : ', timesteps_o[-1])
            # print(x)
            # print('Shape of Tensor x : ',np.shape(x))
            # print('Shape of Tensor x : ',np.shape(x))
            # print('Shape of state : ',np.shape(x_st_t))
            # print('Shape of neighbors_data : ',np.shape(neighbors_data_st))
            # print('Shape of neighbors_edge : ',np.shape(neighbors_edge_value))

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                # print('predictions_dict[ts][nodes[i]]: ',predictions_dict[ts][nodes[i]])
                # print("ts: ", ts)

        return predictions_dict
        
    # def train_loss(self, batch, node_type):
    #     (first_history_index,
    #      x_t, y_t, x_st_t, y_st_t,
    #      neighbors_data_st,
    #      neighbors_edge_value,
    #      robot_traj_st_t,
    #      map) = batch

    #     x = x_t.to(self.device)
    #     y = y_t.to(self.device)
    #     x_st_t = x_st_t.to(self.device)
    #     y_st_t = y_st_t.to(self.device)
    #     if robot_traj_st_t is not None:
    #         robot_traj_st_t = robot_traj_st_t.to(self.device)
    #     if type(map) == torch.Tensor:
    #         map = map.to(self.device)

    #     # Run forward pass
    #     model = self.node_models_dict[node_type]
    #     loss = model.train_loss(inputs=x,
    #                             inputs_st=x_st_t,
    #                             first_history_indices=first_history_index,
    #                             labels=y,
    #                             labels_st=y_st_t,
    #                             neighbors=restore(neighbors_data_st),
    #                             neighbors_edge_value=restore(neighbors_edge_value),
    #                             robot=robot_traj_st_t,
    #                             map=map,
    #                             prediction_horizon=self.ph)

    #     return loss

    # def eval_loss(self, batch, node_type):
    #     (first_history_index,
    #      x_t, y_t, x_st_t, y_st_t,
    #      neighbors_data_st,
    #      neighbors_edge_value,
    #      robot_traj_st_t,
    #      map) = batch

    #     x = x_t.to(self.device)
    #     y = y_t.to(self.device)
    #     x_st_t = x_st_t.to(self.device)
    #     y_st_t = y_st_t.to(self.device)
    #     if robot_traj_st_t is not None:
    #         robot_traj_st_t = robot_traj_st_t.to(self.device)
    #     if type(map) == torch.Tensor:
    #         map = map.to(self.device)

    #     # Run forward pass
    #     model = self.node_models_dict[node_type]
    #     nll = model.eval_loss(inputs=x,
    #                           inputs_st=x_st_t,
    #                           first_history_indices=first_history_index,
    #                           labels=y,
    #                           labels_st=y_st_t,
    #                           neighbors=restore(neighbors_data_st),
    #                           neighbors_edge_value=restore(neighbors_edge_value),
    #                           robot=robot_traj_st_t,
    #                           map=map,
    #                           prediction_horizon=self.ph)

    #     return nll.cpu().detach().numpy()

    
