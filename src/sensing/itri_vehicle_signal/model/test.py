"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Input, Dense, Flatten, Dropout, ZeroPadding3D, Reshape, Activation, GRU
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from collections import deque
import sys
from keras.utils.vis_utils import plot_model
from data import DataSet
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import os.path
import sys


class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
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

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')


        self.model = self.conv2d_gru()

        # self.model = self.attention(use_attention=False)
        # self.model = self.cnn(trainable=False)

        # # Get the appropriate model.
        # if self.saved_model is not None:
        #     print("Loading model %s" % self.saved_model)
        #     self.model = load_model(self.saved_model)
        # elif model == 'lstm':
        #     print("Loading LSTM model.")
        #     self.input_shape = (seq_length, features_length)
        #     self.model = self.lstm()
        # else:
        #     print("Unknown network.")
        #     sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=5e-6, decay=5e-7)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

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
        # Get model with pretrained weights.
        x = self.model.output
        image_shape = (224 , 224, 3)
        input_tensor = Input(image_shape)

        base_model = InceptionV3(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=True
        )

        base_model.model = Model(
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
        # model = Sequential()
        # model.add(LSTM(2048, return_sequences=False,
        #                input_shape=self.input_shape,
        #                dropout=0.5))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.nb_classes, activation='softmax'))
        lstm_model = self.model.output
        lstm_model = LSTM(2048, return_sequences=False,dropout=0.5, name="lstm_1")(lstm_model)
        lstm_model = Dense(512, activation='relu', name="dense_layer_1")(lstm_model)
        lstm_model = Dropout(0.5)(lstm_model)
        lstm_model = Dense(self.nb_classes, activation='softmax', name="dense_layer_2")(lstm_model)

        model = Model(
            inputs=self.model.input,
            outputs=lstm_model
        )

        return model

    def conv2d_gru(self, image_shape=(20, 224, 224, 3)):

        model=Sequential()
        model.add(TimeDistributed(Conv2D(16, kernel_size=(3,3), data_format="channels_last"),input_shape=image_shape))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(Conv2D(16, kernel_size =(3,3))))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        # model.add(TimeDistributed(Reshape((-1,1000))))
        model.add(TimeDistributed(Dense(32)))
        model.add(GRU(512))
        model.add(Dense(self.nb_classes))
        model.add(Activation("softmax"))
        print(model.summary())
        return model


    def extract(self, image_path):
        img = image.load_img(image_path)

        return self.extract_image(img)
    def extract_image(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

def main():
    if (len(sys.argv) == 5):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        image_height = int(sys.argv[3])
        image_width = int(sys.argv[4])
    else:
        print ("Usage: python train.py sequence_length class_limit image_height image_width")
        print ("Example: python train.py 75 2 720 1280")
        exit (1)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = False # pre-load the sequences into memory
    batch_size = 8
    nb_epoch = 1000
    data_type = 'images'
    image_shape = (image_height, image_width, 3)

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
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)


    rm = ResearchModels(class_limit, model, seq_length, saved_model)
    early_stopper = EarlyStopping(patience=15)
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + '-test-' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            #callbacks=[tb, early_stopper, csv_logger, checkpointer],
            callbacks=[early_stopper, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            #callbacks=[tb, early_stopper, csv_logger, checkpointer],
            callbacks=[early_stopper, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

    # print(test.model.summary())

    plot_model(rm.model, expand_nested=True, show_shapes=True, to_file='test.png')


if __name__ == '__main__':
    #np.set_printoptions(threshold=sys.maxsize)
    main()
