% clear all
% close all
% clc

%%    Hank

load('Brake_table.mat');
load('Steer_gain_table.mat');

%%    Ashain
% load Map_data_ITRI_20190918_MAP.mat
load Map_data_ITRI_20191127_MAP_1.mat
% load Map_data_Shalun_20200309_MAP.mat
% load Map_data_Shalun_20200313_MAP.mat
load ENU2SLAMXY_ITRI_v3_016_section.mat
% load ENU2SLAMXY_Shalun_378.mat
% load SLAMXY2ENU_ITRI_025.mat
%UKF para
afa = 1; kapa = 0; beta = 2;
N_xaug = 5;
lambda = afa^2*(N_xaug+kapa)-N_xaug ;
gamma = sqrt(N_xaug+lambda);

weight_0 = lambda/(N_xaug+lambda);
weight_0_cov = lambda/(N_xaug+lambda)+(1-afa^2+beta);
weight_i = 1/(2*(N_xaug+lambda));
weight=[weight_0,weight_0_cov,weight_i];

Ini;

%%   Swot

Ts = 0.01;
t_gap = 3.0;
D_default = 5;
K_epsilon = 1;
Lambda = 0.5;

