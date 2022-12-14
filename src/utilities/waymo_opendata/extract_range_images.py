#!/usr/bin/env python
import os
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
#from waymo_open_dataset.utils import range_image_utils
#from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def extract_range_images_by_frame(frame):
    (range_images, camera_projections, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame))

    frame.lasers.sort(key=lambda laser: laser.name)
    to_grayscale_factor = 255 / 75.0
    for laser in frame.lasers:
        # laser.name(int) is 1, 2, 3, 4, 5
        symbolic_name = open_dataset.LaserName.Name.Name(laser.name)
        range_image = range_images[laser.name][0]  # first lidar return
        # range_image1 = range_images[laser.name][1]  # second lidar return
        range_image_tensor = tf.convert_to_tensor(range_image.data)
        range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
        lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
        range_image_tensor = tf.where(lidar_image_mask, range_image_tensor, tf.ones_like(range_image_tensor) * 1e10)
        range_image_range = range_image_tensor[..., 0]
        # range_image_intensity = range_image_tensor[..., 1]
        # range_image_elongation = range_image_tensor[..., 2]

        # Save only range in gray scale.
        # distance (in float32) is 0~75m, infinite distance is 10000000000.0
        if not os.path.isdir(symbolic_name):
            os.makedirs(symbolic_name)
        ranges = range_image_range.numpy()
        # Use tiff file to store raw ranges
        # since we cannot save float32 as png files.
        filename = "{}/{}_range.tiff".format(symbolic_name, frame.timestamp_micros)
        img = Image.fromarray(ranges.astype(np.float32))
        img.save(filename)
        print("Write {}".format(filename))

        for row in range(ranges.shape[0]):
            for col in range(ranges.shape[1]):
                if ranges[row][col] > 75:
                    ranges[row][col] = 255
                else:
                    ranges[row][col] = ranges[row][col] * to_grayscale_factor
        filename = "{}/{}_grayscale.png".format(symbolic_name, frame.timestamp_micros)
        img = Image.fromarray(ranges.astype(np.uint8))
        img.save(filename)
        print("Write {}".format(filename))


def extra_range_images(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        extract_range_images_by_frame(frame)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    extra_range_images(args.tfrecord_file)


if __name__ == "__main__":
    main()
