# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from data import DataSet
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Activation
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Lambda, Concatenate
from tensorflow.keras.utils import get_custom_objects

# +
import csv
import argparse
from matplotlib import pyplot as plt
import time

from tensorflow_addons.optimizers import RectifiedAdam

tf.keras.optimizers.RectifiedAdam = RectifiedAdam
# -


"""These are the main training settings. Set each before running
this file."""
parser = argparse.ArgumentParser(description="")
parser.add_argument("--seq_length", type=int, default=32,
                    help="the length of a sequence")
parser.add_argument("--class_limit", type=int, default=4,
                    help="how much classes need to clasify")
parser.add_argument("--saved_model", type=str, default="data/checkpoints/lstm-images.003-1.386.hdf5",
                    help="the path of model")
parser.add_argument("--video_file", type=str, default="result_record/rear_demo_video.mp4",
                    help="the path of video that need to clasify")
args = parser.parse_args()

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
get_custom_objects().update({'WarmUp': WarmUp})


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


# -

def tanhexp(x):
    return x * tf.math.tanh(tf.math.exp(x))
get_custom_objects().update({'tanhexp': tanhexp})


def save_result_as_csv(result):
    # The list of column names as mentioned in the CSV file
    headersCSV = ['video_name','flasher','no_signal','turn_left','turn_right','brake', 'times']
    # headersCSV = ['video_name','L','R','brake']
    with open('result_record/rear_dataset_clasify_0920_time_LSTM.csv', 'a', newline='') as f_object:
        dictwriter_object = csv.DictWriter(f_object, fieldnames=headersCSV)
        for i in result :
            # dic = {'video_name':i[0],'L':i[1][0][0],'R':i[1][0][1],'brake':i[2][0][0]}
            # dic = {'video_name':i[0],'brake':i[1][0][0]}
            # dic = {'video_name':i[0],'flasher':i[1][0][0],'no_signal':i[1][0][1],'turn_left':i[1][0][2],'turn_right':i[1][0][3],'brake':i[2][0][0]}
            dic = {'video_name':i[0],'flasher':i[1][0][0],'no_signal':i[1][0][1],'turn_left':i[1][0][2],'turn_right':i[1][0][3], 'times':i[2]}
            dictwriter_object.writerow(dic)
        f_object.close()
    # with open('ncu_dataset_clasify_result_0317.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in result:
    #         print(i[1][0])
    #         writer.writerow(i)

seq_length = args.seq_length
class_limit = args.class_limit
saved_model = args.saved_model
video_file = args.video_file

capture = cv2.VideoCapture(os.path.join(video_file))
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

print("original shape: (%d x %d)" % (width, height))

# exit(0)

fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')

#save all result of mp4 in folder after clasify
clasify_result = []

filename, file_extension = os.path.splitext(video_file)
result_file = filename + "_result.mp4"
video_writer = cv2.VideoWriter(result_file, fourcc, 30, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

width = int(width)
height = int(height)

# get the model.
# extract_model = Extractor(image_shape=(224, 224, 3))
saved_LSTM_model = load_model(saved_model, custom_objects={'tanhexp': tanhexp, 'RMS_Norm': RMS_Norm,'backend': backend})
# saved_LSTM_model = load_model(saved_model, custom_objects = {'WarmUp': WarmUp})
# print(saved_LSTM_model.summary())
saved_LSTM_model._layers = [layer for layer in saved_LSTM_model._layers if not isinstance(layer, dict)]
plot_model(saved_LSTM_model, expand_nested=True, show_shapes=True, to_file='debug_model_clasify.png')

small_frames = False

frames = []
resized_frames = []
frame_count = 0
while True:
    no_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    #print ("no_frame: %d/%d" % (int(no_frame), int(total_frames)))

    if(total_frames < seq_length):
        small_frames = True

    ret, frame = capture.read()

    # Bail out when the video file ends
    if not ret:
        if(small_frames == True):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            ret, frame = capture.read()
        else:
            break

    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

    # Save each frame of the video to a list
    frame_count += 1
    resized_frames.append(preprocess_input(resized_frame))
    frames.append(frame)

    #print ("count=%d, seq_len=%d" % (int(frame_count), int(seq_length)))

    if frame_count < seq_length:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    # For each frame extract feature and prepare it for classification
    # sequence = []
    # for image in resized_frames:
    #     features = extract_model.extract_image(image)
    #     sequence.append(features)

    # Clasify sequence
    s = time.time()
    prediction = saved_LSTM_model.predict(np.expand_dims(resized_frames, axis=0))
    curr_time = (time.time()-s )*1000
    # print(curr_time)
    print(prediction)
#     print(type(prediction))
    # input()
    turn_light_prediction = prediction
    # turn_light_prediction = np.array([[prediction[0], prediction[1]]])
    brake_prediction = prediction
    clasify_result.append([video_file, turn_light_prediction, curr_time])
    # clasify_result.append([video_file, brake_prediction])
    # clasify_result.append([video_file, turn_light_prediction, brake_prediction])
    values = data.print_class_from_prediction(np.squeeze(turn_light_prediction, axis=0))
    # brake_pred="%s: %.2f" % ("brake", brake_prediction[0])

    # Add prediction to frames and write them to new video
    for image in frames:
        for i in range(len(values)):
			# # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.rectangle(image, (10, 10 * i + 22), (85, 10 * i + 30), (0,0,0), -1)
            cv2.putText(image, values[i], (10, 10 * i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, ( 0, 255, 255), lineType=cv2.LINE_AA)
        # cv2.rectangle(image, (10, 10 * 0 + 22), (85, 10 * 0 + 30), (0,0,0), -1)
        # cv2.putText(image, brake_pred, (10, 10 * 0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, ( 0, 255, 255), lineType=cv2.LINE_AA)
        video_writer.write(image)

    frames = []
    resized_frames = []

    if(small_frames == True):
        break

video_writer.release()
save_result_as_csv(clasify_result)
