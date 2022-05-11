"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from tensorflow.keras.utils import to_categorical

import argparse
import tensorlayer as tl
import tensorflow as tf
# from FEQE import enhancement_model
# from FEQE.model import *
# from FEQE.utils import *
# # import FEQE

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

        # # Init FEQE model
        # self.sess, self.t_sr, self.t_lr = self.set_FEQE()

    @staticmethod
    def get_data():
        """Load our data from file."""
        # with open(os.path.join('data', 'pre-train_dataset', 'data_file.csv'), 'r') as fin:
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y1, y2 = [], [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y1.append(self.get_class_one_hot(row[1]))
            y2.append(int(row[4]))

        return np.array(X), [np.array(y1), np.array(y2)]

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type, train_split_percent):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        rand_data_list=list(range(len(train)))
        random.shuffle(rand_data_list)
        cut_point = int(len(train)*train_split_percent)

        data = train if train_test == 'train' else test

        if train_test == 'train' :
            data = [data[i] for i in rand_data_list[:cut_point]]
            print("Creating %s generator with %d samples." % (train_test, len(data)))
        else :
            data = [data[i] for i in rand_data_list[cut_point:]]
            print("Creating %s generator with %d samples." % (train_test, len(data)))


        while 1:
            X, y1, y2 = [], [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # # FEQE maybe can app after this line
                    # # because FEQE also use a sequence as input
                    # sequence = enhancement_model.run(self.sess, self.t_sr, self.t_lr, frames)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)

                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y1.append(self.get_class_one_hot(sample[1]))
                y2.append(int(sample[4]))
            # if train_test == "test":
            #     print(np.array(y2).shape)
            yield np.array(X), [np.array(y1), np.array(y2)]
            # yield np.array(X), np.array(y1)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        # path = os.path.join('data', 'pre-train_dataset', sample[0], sample[1])
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.

        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
        result = []
        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            result.append("%s: %.2f" % (class_prediction[0], class_prediction[1]))

        return result

"""
    def set_FEQE(sef):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--model_path", type=str, default="FEQE/checkpoint/mse_s2/model.ckpt-2000", help="model path")
        parser.add_argument('--save_path', type=str, default='results')
        parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")

        parser.add_argument('--downsample_type', type=str, default='desubpixel')
        parser.add_argument('--upsample_type', type=str, default='subpixel')
        parser.add_argument('--conv_type', type=str, default='default')
        parser.add_argument('--body_type', type=str, default='resnet')
        parser.add_argument('--n_feats', type=int, default=16,
                            help='number of convolution feats')
        parser.add_argument('--n_blocks', type=int, default=20,
                            help='number of residual block if body_type=resnet')
        parser.add_argument('--n_groups', type=int, default=0,
                            help='number of residual group if body_type=res_in_res')
        parser.add_argument('--n_convs', type=int, default=0,
                            help='number of conv layers if body_type=conv')
        parser.add_argument('--n_squeezes', type=int, default=0,
                            help='number of squeeze blocks if body_type=squeeze')

        parser.add_argument('--scale', type=int, default=4)

        args = parser.parse_args()
        #=================Model===================================
        print('Loading FEQE model...')
        t_lr = tf.placeholder('float32', [1, None, None, 3], name='input_image')
        t_hr = tf.placeholder('float32', [1, None, None, 3], name='label_image')

        opt = {
            'n_feats': args.n_feats,
            'n_blocks': args.n_blocks,
            'n_groups': args.n_groups,
            'n_convs': args.n_convs,
            'n_squeezes': args.n_squeezes,
            'downsample_type': args.downsample_type,
            'upsample_type': args.upsample_type,
            'conv_type': args.conv_type,
            'body_type': args.body_type,
            'scale': args.scale
        }
        t_sr = FEQE(t_lr, opt)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        return sess, t_sr, t_lr
"""