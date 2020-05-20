#!/usr/bin/env python
import os
import argparse
import logging
import tensorflow.compat.v1 as tf
import math
import csv
import numpy as np
import itertools
from PIL import Image

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def show_info_by_tfrecord_file(tfrecord_file):
    if not os.path.isfile(tfrecord_file):
        logging.error("Cannot load tfrecord file: %s", tfrecord_file)
        return
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        print(frame.context.stats)
        break


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    show_info_by_tfrecord_file(args.tfrecord_file)

if __name__ == "__main__":
    main()
