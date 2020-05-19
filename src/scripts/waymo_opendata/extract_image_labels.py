#!/usr/bin/env python
# pylint: skip-file
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

LABEL_TYPE_DICT = {
    0: "UNKNOWN".lower(),
    1: "VEHICLE".lower(),
    2: "PEDESTRIAN".lower(),
    3: "SIGN".lower(),
    4: "CYCLIST".lower(),
}


def save_image(data, filename):
    """Save an image."""
    img = Image.fromarray(tf.image.decode_jpeg(data).numpy(), "RGB")
    img.save(filename)
    print("Write {}".format(filename))


def save_2d_bbox_by_frame(frame):
    #(range_images, camera_projections,
    # range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    #    frame)
    #print(frame.context)
    ts = frame.timestamp_micros
    field_names = ["name", "center_x", "center_y", "width", "length", "type", "tracking_id"]
    for index, cam_labels in enumerate(frame.camera_labels):
        camera_name = open_dataset.CameraName.Name.Name(cam_labels.name)
        if not os.path.isdir(camera_name):
            os.makedirs(camera_name)
        filename = "{}/{}_bbox.csv".format(camera_name, ts)
        with open(filename, "w") as _fp:
            writer = csv.writer(_fp)
            writer.writerow(field_names)
            for label in cam_labels.labels:
                _type = LABEL_TYPE_DICT.get(label.type, "undefined")
                row = [camera_name, label.box.center_x, label.box.center_y, label.box.width, label.box.length, _type, label.id]
                writer.writerow(row)
        print("Write {}".format(filename))


def save_2d_bbox_by_tfrecord_file(tfrecord_file):
    if not os.path.isfile(tfrecord_file):
        logging.error("Cannot load tfrecord file: %s", tfrecord_file)
        return
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        save_2d_bbox_by_frame(frame)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    save_2d_bbox_by_tfrecord_file(args.tfrecord_file)

if __name__ == "__main__":
    main()
