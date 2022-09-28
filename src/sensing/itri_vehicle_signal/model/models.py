# -*- coding: utf-8 -*-
"""
A collection of models we'll use to attempt to classify videos.
"""
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Add, Multiply, Reshape, Activation
from tensorflow.keras.layers import ZeroPadding2D, ZeroPadding3D, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Lambda, Concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, schedules
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from tensorflow.keras.layers import ConvLSTM2D
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
# from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.python.keras.activations import tanh, exponential
from collections import deque
import sys
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import get_custom_objects
from data import DataSet
import numpy as np
import argparse
import random

from matplotlib import pyplot as plt

from WaveNetClassifier import WaveNetClassifier

# # +

# # Seed value
# # Apparently you may use different seed values at each stage
# seed_value= 1
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)
# # 3. Set the `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)

# # -


from typing import Callable
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


# +
class RMS_Norm(LayerNormalization):
    def __init__(self, eps=1e-8, p=-1., bias=False, scope=None, name=None, *args, **kwargs):
        super(RMS_Norm, self).__init__(*args, **kwargs)
        """
            Root Mean Square Layer Normalization
        :param x: input tensor, with shape [batch, ..., dimension]
        :param eps: epsilon value, default 1e-8
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        :param scope: the variable scope
        :return: a normalized tensor, with shape as `inputs`
        """

        self._name = name
        self.eps = eps
        self.p = p
        self.bias = bias
        self.scope = scope      
     
    def call(self, inputs):
        
        layer_size = inputs.get_shape().as_list()[-1]

        self.gamma = tf.ones([layer_size])
        if self.bias:
            self.beta = tf.zeros([layer_size])
        else:
            self.beta = 0
        if self.p < 0. or self.p > 1.:
            ms = tf.reduce_mean(inputs ** 2, -1, keepdims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(inputs, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keepdims=True)

        # ms = tf.reduce_sum(tf.square(inputs), axis=-1,
        #                    keep_dims=True) * 1./self.layer_size

        norm_inputs = inputs * tf.math.rsqrt(ms + self.eps)
        return tf.multiply(self.gamma, norm_inputs) + self.beta
    
get_custom_objects().update({'RMS_Norm': RMS_Norm})


# +
class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048, image_shape=(224,224,3)):
        """
        `model` = lstm (only one for this case)
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.image_shape = image_shape
        self.input_shape = (seq_length,)+image_shape

        # Set the metrics. Only use top k if there's a need.
        metrics = ['categorical_accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        self.model = self.attention(use_attention=False)
        self.model = self.cnn(trainable=True)

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            # self.model = load_model(self.saved_model)
            # for layer in self.model.layers:
            #     layer.trainable = True           
            self.model = self.lstm()
            self.model.load_weights(self.saved_model, by_name = True)
        elif model == 'lstm':
            print("Loading LSTM model.")
            # self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.

        lr_schedule = schedules.ExponentialDecay(initial_learning_rate=8e-5,
                                                decay_steps=3678,#4462#2448
                                                decay_rate=0.9)
        wd_schedule = schedules.ExponentialDecay(initial_learning_rate=1e-7,
                                                decay_steps=3678,
                                                decay_rate=0.99)
#         cosine_lr = tf.keras.experimental.CosineDecay(
#                         initial_learning_rate=3e-4, decay_steps=2157)
#         cosine_wd = tf.keras.experimental.CosineDecay(
#                         initial_learning_rate=3e-1, decay_steps=2157)
#         optimizer = Adam(learning_rate=lr_schedule)
        # optimizer = RectifiedAdam(learning_rate=lr_schedule, weight_decay=wd_schedule)
#         optimizer = RectifiedAdam(learning_rate=cosine_lr, weight_decay=cosine_wd)
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
#         optimizer = Adam(lr=3e-4, decay=5e-8)
#         optimizer = Adam(lr=3e-5)
        # optimizer = Adam(decay=5e-8)
        alpha = 0.6
        self.model.compile(loss='categorical_crossentropy',
                                # {'turnlight_output': 'categorical_crossentropy',
                                #  'avg_brake_output': 'binary_crossentropy'},
                           # loss_weights = {'turnlight_output': alpha,'avg_brake_output': 1-alpha},
                           optimizer=optimizer,
                           metrics= 'categorical_accuracy')#tf.keras.metrics.AUC()
                                # {'turnlight_output': 'categorical_accuracy',
                                #  'avg_brake_output': 'binary_accuracy'})

        print(self.model.summary())

    def tanhexp(x):
        return tf.multiply(x,tf.tanh(tf.exp(x)))
    get_custom_objects().update({'tanhexp': tanhexp})

    def attention(self, use_attention=False):
        input_tensor = Input(self.input_shape, name="img_input")
        model = Model(
            inputs=input_tensor,
            outputs=input_tensor
        )
        # print(model.summary())
        return model

    def cnn(self, trainable = False):
        # Get model with pretrained weights.

        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3

        def channel_attention(x, reduction_ratio=0.125):
            channel = int(x.shape[channel_axis])
            maxpool_channel = GlobalMaxPooling2D()(x)
            maxpool_channel = Reshape((1,1,channel))(maxpool_channel)
            avgpool_channel = GlobalAveragePooling2D()(x)
            avgpool_channel = Reshape((1,1,channel))(avgpool_channel)
            Dense_One = Dense(units = int(channel * reduction_ratio), activation = 'relu', kernel_initializer='he_normal', use_bias = True, bias_initializer = 'zeros')
            Dense_Two = Dense(units = int(channel), activation = 'relu', kernel_initializer='he_normal', use_bias = True, bias_initializer = 'zeros')

            mlp_1_max = Dense_One(maxpool_channel)
            mlp_2_max = Dense_Two(mlp_1_max)
            mlp_2_max = Reshape(target_shape = (1,1, int(channel)))(mlp_2_max)

            mlp_1_avg = Dense_One(avgpool_channel)
            mlp_2_avg = Dense_Two(mlp_1_avg)            
            mlp_2_avg = Reshape(target_shape = (1,1, int(channel)))(mlp_2_avg)

            channel_attention_feature = Add()([mlp_2_max,mlp_2_avg])
            channel_attention_feature = Activation('sigmoid')(channel_attention_feature)

            return Multiply()([channel_attention_feature, x])

        def spatial_attention(channel_refined_feature):
            maxpool_spatial = Lambda(lambda x :backend.max(x, axis=3, keepdims=True))(channel_refined_feature)
            avgpool_spatial = Lambda(lambda x :backend.mean(x, axis=3, keepdims=True))(channel_refined_feature)
            max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])

            return Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid', kernel_initializer='he_normal',use_bias=False)(max_avg_pool_spatial)

        def cbam_module(x, reduction_ratio=0.5):
            channel_refined_feature = channel_attention(x,reduction_ratio=reduction_ratio)
            spatial_attention_feature = spatial_attention(channel_refined_feature)
            refined_feature = Multiply()([channel_refined_feature, spatial_attention_feature])

            return  refined_feature
        
        def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None, drop=0.5):
            """A residual block.

            Args:
                x: input tensor.
                filters: integer, filters of the bottleneck layer.
                kernel_size: default 3, kernel size of the bottleneck layer.
                stride: default 1, stride of the first layer.
                conv_shortcut: default False, use convolution shortcut if True,
                otherwise identity shortcut.
                name: string, block label.

            Returns:
            Output tensor for the residual block.
            """
            bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

            # preact = LayerNormalization(
            #     axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_ln')(x)
            # preact = RMS_Norm(name=name + '_preact_rmsn')(x)
            preact = x
            preact = Activation('tanhexp', name=name + '_preact_tanhexp')(preact)
            preact = Dropout(drop)(preact)

            if conv_shortcut:
                shortcut = Conv2D(
                        4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
            else:
                shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

            x = Conv2D(
                  filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
            # x = LayerNormalization(
            #       axis=bn_axis, epsilon=1.001e-5, name=name + '_1_ln')(x)
            # x = RMS_Norm(name=name + '_1_rmsn')(x)
            x = Activation('tanhexp', name=name + '_1_tanhexp')(x)
            x = Dropout(drop)(x)

            x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
            x = Conv2D(
                  filters,
                  kernel_size,
                  strides=stride,
                  use_bias=False,
                  name=name + '_2_conv')(x)
            # x = LayerNormalization(
            #       axis=bn_axis, epsilon=1.001e-5, name=name + '_2_ln')(x)
            # x = RMS_Norm(name=name + '_2_rmsn')(x)
            x = Activation('tanhexp', name=name + '_2_tanhexp')(x)
            x = Dropout(drop)(x)

            x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)


#             x = cbam_module(x)

            x = Add(name=name + '_out')([shortcut, x])

#             x = Activation('relu', name=name + '_3_relu')(x)
            return x
        
        def stack2(x, filters, blocks, stride1=2, name=None, drop1=0.33):
            """A set of stacked residual blocks.

            Args:
                x: input tensor.
                filters: integer, filters of the bottleneck layer in a block.
                blocks: integer, blocks in the stacked blocks.
                stride1: default 2, stride of the first layer in the first block.
                name: string, stack label.

            Returns:
                Output tensor for the stacked blocks.
            """
            x = block2(x, filters, conv_shortcut=True, name=name + '_block1', drop=drop1)
            for i in range(2, blocks):
                x = block2(x, filters, name=name + '_block' + str(i), drop=drop1)
            x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks), drop=drop1)
            return x
        
        def stack_fn(x):
            x = stack2(x, 64, 3, name='conv2')
            x = cbam_module(x)
            x = stack2(x, 128, 2, name='conv3')
            # x = cbam_module(x)
            # x = stack2(x, 256, 2, name='conv4')
            return x
            # return stack2(x, 512, 3, stride1=1, name='conv5')

        # Get model with pretrained weights.
        x = self.model.output
        input_tensor = Input(self.image_shape)

        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            pooling="max",
            input_tensor=input_tensor,
        )
        # base_model = Model(
        #     inputs=input_tensor,
        #     outputs=base_model.get_layer('block5_pool').output
        #     # outputs=turnlight_fe
        #     # outputs=[turnlight_fe, brakelight_fe]
        #     # outputs=[Lturnlight_fe, Rturnlight_fe, brakelight_fe]
        # )

        # for layer in base_model.layers:
        #     print(layer)
        # Block 5
        turnlight_fe = Conv2D(512, (3, 3), activation='relu', padding='same', name='Tblock5_conv1')(base_model.get_layer('block4_pool').output)
        turnlight_fe = Conv2D(512, (3, 3), activation='relu', padding='same', name='Tblock5_conv2')(turnlight_fe)
        turnlight_fe = Conv2D(512, (3, 3), activation='relu', padding='same', name='Tblock5_conv3')(turnlight_fe)
        turnlight_fe = MaxPooling2D((2, 2), strides=(2, 2), name='Tblock5_pool')(turnlight_fe)
        turnlight_fe = GlobalMaxPooling2D()(turnlight_fe)


        base_model = Model(
            inputs=input_tensor,
            outputs=turnlight_fe
            # outputs=turnlight_fe
            # outputs=[turnlight_fe, brakelight_fe]
            # outputs=[Lturnlight_fe, Rturnlight_fe, brakelight_fe]
        )

        # base_output = []
        # for out in base_model.output:
        #     base_output.append(TimeDistributed(Model(base_model.input, out))(x))
        # output1, output2 = base_output 
        
        base_model = TimeDistributed(base_model, name='td_feature_extractor')(x)
       
        
        # We'll extract features at the final pool layer.
        model = Model(
            inputs=self.model.input,
            outputs=base_model
            # outputs = [output1, output2]
        )

        # model.trainable = False 
        # print(model.summary())
        return model
    
    def lstm(self):
        # """Build a simple LSTM network. We pass the extracted features from
        # our CNN to this model predomenently."""
        # # Model.

        """
        # Turn light detection
        turnlight_model = self.model.output
        turnlight_model = LSTM(2048, return_sequences=False,dropout=0.5, name="turnlight_lstm_1")(turnlight_model)
        # turnlight_model = Dense(1024, activation='relu', name="turnlight_dense_layer_1")(turnlight_model)
        # turnlight_model = Dropout(0.5)(turnlight_model)
        turnlight_model = Dense(512, activation='relu', name="turnlight_dense_layer_2")(turnlight_model)
        turnlight_model = Dropout(0.5)(turnlight_model)
        # turnlight_model = Dense(128, activation='relu', name="turnlight_dense_layer_3")(turnlight_model)
        # turnlight_model = Dropout(0.5)(turnlight_model)
        turnlight_model = Dense(self.nb_classes, activation='softmax', name="turnlight_output")(turnlight_model)
        """


        x = self.model.output
        # x = Concatenate(axis= -2)([x, x, x, x])
        wavenet_class = WaveNetClassifier(input_tensor=x, output_shape=self.nb_classes, kernel_size = 2, dilation_depth = 4, n_filters = 512, task = 'classification')
        # wavenet_L_class = WaveNetClassifier(input_tensor=L, output_shape=1, kernel_size = 2, dilation_depth = 5, n_filters = 128, task = 'regression',regression_range=[0, 1])
        turnlight_model = wavenet_class.get_model()


        # Brake detection
        # brake_model = self.model.output    
        # wavenet_B_class = WaveNetClassifier(input_tensor=x, output_shape=1, kernel_size = 2, dilation_depth = 5, n_filters = 512, task = 'regression',regression_range=[0, 1],name='B')
        # brake_model = wavenet_B_class.get_model()
        # # brake_model = TimeDistributed(Dense(1024, activation='relu', name="brake_dense_layer_1"))(brake_model)
        # # brake_model = TimeDistributed(Dropout(0.5))(brake_model)
        # brake_model = TimeDistributed(Dense(512, activation='relu', name="brake_dense_layer_2"))(brake_model)
        # # brake_model = TimeDistributed(Dropout(0.3))(brake_model)
        # brake_model = TimeDistributed(Dense(64, activation='relu', name="brake_dense_layer_3"))(brake_model)
        # brake_model = TimeDistributed(Dense(1, activation='sigmoid', name="brake_output"))(brake_model)
        # brake_model = GlobalAveragePooling1D(name="avg_brake_output")(brake_model)


        model = Model(
            inputs=self.model.input,
            # outputs=[turnlight_model, brake_model]
            outputs=turnlight_model
            # outputs=brake_model
        )
        return model


# +
def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seq_length", type=int, default=32,
                        help="the length of a sequence")
    parser.add_argument("--class_limit", type=int, default=4,
                        help="how much classes need to clasify")
    parser.add_argument("--image_height", type=int, default=224,
                        help="how to reszie the image height")
    parser.add_argument("--image_width", type=int, default=224,
                        help="how to reszie the image width")
    args = parser.parse_args()

    seq_length = args.seq_length
    class_limit = args.class_limit
    image_height = args.image_height
    image_width = args.image_width


    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    # saved_model = './data/checkpoints/lstm-images.005-0.292.hdf5'
    load_to_memory = False # pre-load the sequences into memory

    batch_size = 1
    nb_epoch = 1000
    data_type = 'images'
    image_shape = (image_height, image_width, 3)
    train_split_percent=0.1

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
            )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
            )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    # This needs to be modified, otherwise the data will be reused between training and valid
    train_data, test_data = data.split_train_test()
    steps_per_epoch = (len(train_data) * train_split_percent * 0.1) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, train_split_percent)
        val_generator = data.frame_generator(batch_size, 'test', data_type, train_split_percent)


    test = ResearchModels(class_limit, model, seq_length, saved_model, image_shape=image_shape)

    # print(test.model.summary())

    # print(len(test.model.layers))

    H = test.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=1,
            # callbacks=[tb, early_stopper, csv_logger, checkpointer],
            # callbacks=[early_stopper, checkpointer],
            validation_data=val_generator,
            validation_steps=2,
            workers=4)

    test.model._layers = [layer for layer in test.model._layers if not isinstance(layer, dict)]
    plot_model(test.model, expand_nested=True, show_shapes=True, to_file='model_struct.png')
# -


if __name__ == '__main__':
    #np.set_printoptions(threshold=sys.maxsize)
    main()
