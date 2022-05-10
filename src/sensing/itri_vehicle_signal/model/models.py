"""
A collection of models we'll use to attempt to classify videos.
"""
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, ZeroPadding3D, Reshape, AveragePooling1D
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from collections import deque
import sys
from tensorflow.python.keras.utils.vis_utils import plot_model
from data import DataSet
import numpy as np
import argparse
import random

from WaveNetClassifier import WaveNetClassifier


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

        # Set the metrics. Only use top k if there's a need.
        metrics = ['categorical_accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        self.model = self.attention(use_attention=False)
        self.model = self.cnn(trainable=False)

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            # self.model = load_model(self.saved_model)
            self.model = self.lstm()
            self.model.load_weights(self.saved_model, by_name = True)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        # optimizer = Adam(lr=5e-6, decay=5e-7)
        optimizer = Adam(lr=1e-5, decay=5e-7)
        self.model.compile(loss=#'categorical_crossentropy',
                                {'turnlight_output': 'categorical_crossentropy',
                                 'average_brake_output': 'binary_crossentropy'},
                           loss_weights = {'turnlight_output': 1.0, 'average_brake_output': 0.25},
                           optimizer=optimizer,
                           metrics= #'categorical_accuracy')
                                {'turnlight_output': 'accuracy',
                                 'average_brake_output': 'accuracy'})

        print(self.model.summary())

    def attention(self, use_attention=False, image_shape=(20, 224, 224, 3)):
        input_tensor = Input(image_shape, name="img_input")
        if use_attention:
            # Get model with pretrained weights.
            base_model = TimeDistributed(Conv2D(32, (3, 3), dilation_rate=1, strides=(1, 1), padding="same", name="attention_conv1"))(input_tensor)
            base_model = TimeDistributed(Conv2D(64, (3, 3), dilation_rate=2, strides=(1, 1), padding="same", name="attention_conv2"))(base_model)
            base_model = TimeDistributed(Conv2D(64, (3, 3), dilation_rate=2, strides=(1, 1), padding="same", name="attention_conv3"))(base_model)
            base_model = TimeDistributed(Conv2D(3, (3, 3), dilation_rate=1, strides=(1, 1), padding="same", name="attention_conv4"))(base_model)

            # base_model.trainable = True   # todo: find-tune the model

            model = Model(
                inputs=input_tensor,
                outputs=base_model
            )
        else :
            model = Model(
                inputs=input_tensor,
                outputs=input_tensor
            )
        # print(model.summary())
        return model

    def cnn(self, trainable = False):
        # # Get model with pretrained weights.
        # x = self.model.output

        # base_model = InceptionV3(
        #     weights='imagenet',
        #     include_top=True
        # )

        # base_model.model = Model(
        #     inputs=base_model.input,
        #     outputs=base_model.get_layer('avg_pool').output
        # )


        # Get model with pretrained weights.
        x = self.model.output
        # image_shape = (224 , 224, 3)
        input_tensor = Input(self.image_shape)


        base_model = ResNet50(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )


        base_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

        # base_model.trainable = True   # todo: find-tune the model
        # print(base_model.summary())
        base_model = TimeDistributed(base_model, name='td_feature_extractor')(x)



        # base_model = Flatten()(base_model)
        # shape = base_model.shape
        # print(shape)
        # print((shape[1], shape[2]*shape[3]*shape[4]))
        # base_model = Reshape((shape[1], shape[2]*shape[3]*shape[4]))(base_model)
        
        # We'll extract features at the final pool layer.
        model = Model(
            inputs=self.model.input,
            outputs=base_model
        )

        #Test freeze cnn part
        for layer in model.layers:
            layer.trainable = trainable

        # model.trainable=False
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
        turnlight_model = Dense(1024, activation='relu', name="turnlight_dense_layer_1")(turnlight_model)
        turnlight_model = Dropout(0.5)(turnlight_model)
        turnlight_model = Dense(512, activation='relu', name="turnlight_dense_layer_2")(turnlight_model)
        turnlight_model = Dropout(0.5)(turnlight_model)
        turnlight_model = Dense(128, activation='relu', name="turnlight_dense_layer_3")(turnlight_model)
        turnlight_model = Dropout(0.5)(turnlight_model)
        turnlight_model = Dense(self.nb_classes, activation='softmax', name="turnlight_output")(turnlight_model)
        """
        x = self.model.output
        wavenet_class = WaveNetClassifier(input_tensor=x, output_shape=self.nb_classes, kernel_size = 2, dilation_depth = 9, task = 'classification')
        turnlight_model = wavenet_class.get_model()
        # turnlight_model = Model(
        #     inputs=self.model.output,
        #     outputs=turnlight_model
        # )


        # Brake detection
        brake_model = self.model.output
        # brake_model = LSTM(2048, return_sequences=False, dropout=0.5, name="brake_lstm_1")(brake_model)
        # brake_model = Dense(512, activation='relu', name="brake_dense_layer_1")(brake_model)
        # brake_model = Dropout(0.5)(brake_model)
        # brake_model = Dense(256, activation='relu', name="brake_dense_layer_2")(brake_model)
        # brake_model = Dropout(0.5)(brake_model)
        # brake_model = Dense(1, activation='sigmoid', name="brake_output")(brake_model)
        brake_model = TimeDistributed(Dense(512, activation='relu', name="brake_dense_layer_1"))(brake_model)
        brake_model = TimeDistributed(Dropout(0.5))(brake_model)
        brake_model = TimeDistributed(Dense(128, activation='relu', name="brake_dense_layer_2"))(brake_model)
        brake_model = TimeDistributed(Dropout(0.5))(brake_model)
        brake_model = TimeDistributed(Dense(1, activation='sigmoid', name="brake_output"))(brake_model)
        brake_model = AveragePooling1D(pool_size = self.seq_length, padding='valid', name="average_brake_output")(brake_model)


        model = Model(
            inputs=self.model.input,
            outputs=[turnlight_model, brake_model]
            # outputs=turnlight_model
        )

        return model

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seq_length", type=int, default=20,
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
    # saved_model = './data/checkpoints_0413_with_two_output/lstm-images.016-1.015.hdf5'
    load_to_memory = False # pre-load the sequences into memory
    batch_size = 1
    nb_epoch = 1000
    data_type = 'images'
    image_shape = (image_height, image_width, 3)
    train_split_percent=0.7

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
    steps_per_epoch = (len(train_data) * 0.3) // batch_size


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

    # test.model.fit_generator(
    #         generator=generator,
    #         steps_per_epoch=steps_per_epoch,
    #         epochs=1,
    #         verbose=1,
    #         # callbacks=[tb, early_stopper, csv_logger, checkpointer],
    #         # callbacks=[early_stopper, checkpointer],
    #         validation_data=val_generator,
    #         validation_steps=2,
    #         workers=4)

    test.model._layers = [layer for layer in test.model._layers if not isinstance(layer, dict)]
    plot_model(test.model, expand_nested=True, show_shapes=True, to_file='model_struct.png')


if __name__ == '__main__':
    #np.set_printoptions(threshold=sys.maxsize)
    main()
