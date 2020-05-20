#!/usr/bin/env python
import os
import argparse
import logging
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from PIL import Image

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset



def save_image(data, filename):
    """Save an image."""
    img = Image.fromarray(tf.image.decode_jpeg(data).numpy(), "RGB")
    img.save(filename)
    print("Write {}".format(filename))


def save_images_by_frame(frame):
    #(range_images, camera_projections,
    # range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    #    frame)
    # print(frame.context)
    ts = frame.timestamp_micros
    for index, image in enumerate(frame.images):
        camera_name = open_dataset.CameraName.Name.Name(image.name)
        if not os.path.isdir(camera_name):
            os.makedirs(camera_name)
        image_filename = "{}/{}.jpg".format(camera_name, ts)
        save_image(image.image, image_filename)


def save_images_by_tfrecord_file(tfrecord_file):
    if not os.path.isfile(tfrecord_file):
        logging.error("Cannot load tfrecord file: %s", tfrecord_file)
        return
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        save_images_by_frame(frame)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    save_images_by_tfrecord_file(args.tfrecord_file)

if __name__ == "__main__":
    main()
