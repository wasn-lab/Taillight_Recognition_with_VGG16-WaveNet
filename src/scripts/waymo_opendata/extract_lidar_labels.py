#!/usr/bin/env python
import os
import argparse
import logging
import csv
import tensorflow.compat.v1 as tf

#from waymo_open_dataset.utils import range_image_utils
#from waymo_open_dataset.utils import transform_utils
#from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

LABEL_TYPE_DICT = {
    0: "UNKNOWN".lower(),
    1: "VEHICLE".lower(),
    2: "PEDESTRIAN".lower(),
    3: "SIGN".lower(),
    4: "CYCLIST".lower(),
}


def save_lidar_labels_by_frame(frame):
    #(range_images, camera_projections,
    # range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    #    frame)
    #print(frame.context)
    field_names = ["center_x", "center_y", "center_z",
                   "width", "length", "height", "heading",
                   "speed_x", "speed_y",
                   "accel_x", "accel_y",
                   "type", "tracking_id", "num_lidar_points_in_box"]
    filename = "{}_lidar_labels.csv".format(frame.timestamp_micros)
    with open(filename, "w") as _fp:
        writer = csv.writer(_fp)
        writer.writerow(field_names)
        for label in frame.laser_labels:
            box = label.box
            row = [box.center_x, box.center_y, box.center_z,
                   box.width, box.length, box.height, box.heading,
                   label.metadata.speed_x, label.metadata.speed_y,
                   label.metadata.accel_x, label.metadata.accel_y,
                   LABEL_TYPE_DICT.get(label.type, "undefined"),
                   label.id,
                   label.num_lidar_points_in_box]
            writer.writerow(row)
        print("Write {}".format(filename))


def save_lidar_labels_by_tfrecord_file(tfrecord_file):
    if not os.path.isfile(tfrecord_file):
        logging.error("Cannot load tfrecord file: %s", tfrecord_file)
        return
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        save_lidar_labels_by_frame(frame)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    save_lidar_labels_by_tfrecord_file(args.tfrecord_file)

if __name__ == "__main__":
    main()
