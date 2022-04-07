import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import math
from FEQE.model.desubpixel import DeSubpixelConv2d
import numpy as np
import scipy.io

def init(in_feats, kernel_size=3):
    std = 1./math.sqrt(in_feats*(kernel_size**2))
    return tf.random_uniform_initializer(-std, std)

class NormalizeLayer(Layer):
    def __init__(self, prev_layer, mean, std, name='normalize_layer'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs
        self.outputs = (self.inputs - mean)*std

        self.all_layers.append(self.outputs)

class RestoreLayer(Layer):
    def __init__(self,prev_layer, mean, std, name='restore_layer'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs
        self.outputs = (self.inputs/std) + mean

        self.all_layers.append(self.outputs)

class Bicubic(Layer):
    def __init__(self,prev_layer, name='bicubic'):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        shape = tf.shape(self.inputs)
        h, w = shape[1], shape[2]
        self.outputs = tf.image.resize_images(self.inputs, [h//4, w//4], tf.image.ResizeMethod.BICUBIC)
        self.all_layers.append(self.outputs)

def conv(x, in_feats, out_feats, kernel_sizes=(3, 3), strides=(1, 1), act=None, conv_type='default', name='conv'):
    with tf.variable_scope(name):
        if conv_type == 'default':
            x = Conv2d(x, out_feats, kernel_sizes, strides, act=act, 
                       W_init=init(in_feats, kernel_sizes[0]), 
                       b_init=init(in_feats, kernel_sizes[0]))

        elif conv_type == 'depth_wise':
            x = DepthwiseConv2d(x, kernel_sizes, strides, act=tf.nn.relu, W_init=init(in_feats), 
                                b_init=init(in_feats), name='depthwise')
            x = Conv2d(x, out_feats, (1, 1), (1, 1), act=act, 
                       W_init=init(in_feats, kernel_sizes[0]), 
                       b_init=init(in_feats, kernel_sizes[0]), name='conv')

        else:
            raise Exception('Unknown conv type', conv_type)
    return x

def downsample(x, n_feats, scale=4, conv_type='default', sample_type='subpixel', name='downsample'):
    with tf.variable_scope(name):
        if sample_type == 'desubpixel':
            assert scale == 2 or scale == 4
            if scale == 2:
                x = conv(x, 3, n_feats//4, (1, 1), act=None, conv_type=conv_type, name='conv')
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle')
            else:
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle1')
                x = conv(x, 12, n_feats//4, (1, 1), act=None, conv_type=conv_type, name='conv2')
                x = DeSubpixelConv2d(x, 2, name='pixel_deshuffle2')

        elif sample_type == 'conv_s2':
            x = conv(x, 3, n_feats, (1, 1), strides=(2, 2), act=tf.nn.relu, name='conv1_stride2')
            x = conv(x, n_feats, n_feats, (1, 1), strides=(2, 2), act=tf.nn.relu, name='conv2_stride2')

        elif sample_type == 'bicubic':
            x = RestoreLayer(x, 0.5, 255)
            x = Bicubic(x)
            x = NormalizeLayer(x, 0.5, 255)
            x = conv(x, 3, n_feats, (1, 1), act=tf.nn.relu, name='conv1')

        elif sample_type == 'pooling':
            x = MaxPool2d(x, (2, 2))
            x = conv(x, 12, n_feats, (1, 1), act=None, conv_type=conv_type, name='conv')
            x = MaxPool2d(x, (2, 2))

        elif sample_type == 'none':
            x = conv(x, 3, n_feats, act=tf.nn.relu, name='conv')

        else: 
            raise Exception('Unknown sample_type', sample_type)
    return x

def upsample(x, n_feats, scale=4, conv_type='default', sample_type='subpixel', name='upsample'):
    with tf.variable_scope(name):
        if sample_type == 'subpixel':
            assert scale == 2 or scale == 4
            if scale == 2:
                x = conv(x, n_feats, 3*4, (1, 1), act=None, conv_type=conv_type, name='conv')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle')
            else:
                x = conv(x, n_feats, n_feats*4, (1, 1), act=None, conv_type=conv_type, name='conv1')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle1')# /1
                x = conv(x, n_feats, 3*4, (1, 1), act=None, conv_type=conv_type, name='conv2')
                x = SubpixelConv2d(x, scale=2, n_out_channel=None, name='pixelshuffle2')

        elif sample_type == 'deconv':
            x = DeConv2d(x, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())
            x = DeConv2d(x, n_feats, n_feats, strides=(2, 2), act=tf.nn.relu, W_init=init(), b_init=init())

        elif sample_type == 'none':
            x = conv(x, n_feats, n_feats, act=tf.nn.relu, name='conv')

        else:
            raise Exception('Unknown sample_type', sample_type)
    return x

def res_block(x, n_feats, conv_type='default', name='res_block'):
    with tf.variable_scope(name):
        res = conv(x, n_feats, n_feats, act=tf.nn.relu, conv_type=conv_type, name='conv1')
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='conv2')
        x = ElementwiseLayer([x, res], tf.add, name='res_add')
    return x

def res_group(x, n_feats, n_blocks, conv_type='default', name='res_group'):
    with tf.variable_scope(name):
        res = x
        for i in range(n_blocks):
            res = res_block(res, n_feats, conv_type=conv_type, name='res_block%d' %i)
        x = ElementwiseLayer([x, res], tf.add, name='add')
    return x

def fire(x, n_feats, conv_type, name):
    with tf.variable_scope(name):
        res = conv(x, n_feats, n_feats//4, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv1')
        res_11 = conv(res, n_feats//4, n_feats//2, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv2')
        res_33 = conv(res, n_feats//4, n_feats//2, act=tf.nn.relu, conv_type=conv_type, name='conv3')
        res = ConcatLayer([res_11, res_33], 3, name='concat1')

        res = conv(x, n_feats, n_feats//4, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv4')
        res_11 = conv(res, n_feats//4, n_feats//2, (1, 1), act=tf.nn.relu, conv_type=conv_type, name='conv5')
        res_33 = conv(res, n_feats//4, n_feats//2, act=tf.nn.relu, conv_type=conv_type, name='conv6')
        res = ConcatLayer([res_11, res_33], 3, name='concat2')
        x = ElementwiseLayer([x, res], tf.add, name='add')

    return x

def body(res, n_feats, n_groups, n_blocks, n_convs, n_squeezes, body_type='resnet', conv_type='default', name='body'):
    with tf.variable_scope(name):
        if body_type == 'resnet':
            for i in range(n_blocks):
                res = res_block(res, n_feats, conv_type=conv_type, name='res_block%d' %i)
        elif body_type == 'res_in_res':
            for i in range(n_groups):
                res = res_group(res, n_feats, n_blocks, conv_type=conv_type, name='res_group%d' %i)
        elif body_type == 'conv':
            for i in range(n_convs):
                res = conv(res, n_feats, n_feats, conv_type=conv_type, name='conv%d' %i)
        elif body_type == 'squeeze':
            for i in range(n_squeezes):
                res = fire(res, n_feats, conv_type, name='fire%d' %i) 
        else:
            raise Exception('Unknown body type', body_type)
        
        res = conv(res, n_feats, n_feats, act=None, conv_type=conv_type, name='res_lastconv')
    return res

def FEQE(t_bicubic, opt):

    #############Option Mutual Exclusive###############
    # body_type=resnet:         n_blocks is required
    # body_type='res_in_res':   n_blocks and n_groups are required
    # body_type='conv':         n_convs is required
    # body_type='squeeze':      n_squeezes is required
    
    downsample_type = opt['downsample_type']
    upsample_type = opt['upsample_type'] 
    conv_type   = opt['conv_type']
    body_type   = opt['body_type']

    n_feats     = opt['n_feats']
    n_blocks    = opt['n_blocks']
    n_groups    = opt['n_groups']
    n_convs     = opt['n_convs']
    n_squeezes  = opt['n_squeezes']

    scale       = opt['scale']

    with tf.variable_scope('Generator') as vs:
        x = InputLayer(t_bicubic, name='in')
        x = NormalizeLayer(x, 0.5, 255)
        g_skip = x

        #===========Downsample==============
        x = downsample(x, n_feats, scale, conv_type, downsample_type)

        #============Residual=================
        x = body(x, n_feats, n_groups, n_blocks, n_convs, n_squeezes, body_type, conv_type)
        
        #=============Upsample==================
        x = upsample(x, n_feats, scale, conv_type, upsample_type)

        x = ElementwiseLayer([x, g_skip], tf.add, name='add_global_res')

        x = RestoreLayer(x, 0.5, 255)
        outputs = tf.clip_by_value(x.outputs, 0, 1)

        return outputs



