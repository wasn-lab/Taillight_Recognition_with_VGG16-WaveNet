#!/usr/bin/env python
# pylint: skip-file
import os
import argparse
import logging
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from PIL import Image
import matplotlib.pyplot as plt

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def color_func(r):
    """Generates a color based on range.

    Args:
    r: the range value of a given point.
    Returns:
    The color for a given range
    """
    # c is tuple like (0.0, 0.0, 0.749554367201426, 1.0)
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)

    return tuple(int(_ * 255) for _ in c[0:3])


def plot_points_on_image(projected_points, camera_image, filename, color_func):
    """Plots points on a camera image.

    Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    filename: output file name
    color_func: a function that generates a color from a range value.
    """
    nums = tf.image.decode_jpeg(camera_image.image).numpy()
    width, height = nums.shape[1], nums.shape[0]
    img = Image.fromarray(nums, "RGB")

    pixel_data = img.load()
    for point in projected_points:
        c = color_func(point[2])
        x = int(point[0])
        y = int(point[1])

        for xoffset in [-1, 0, 1]:
            for yoffset in [-1, 0, 1]:
                xpos = x + xoffset
                ypos = y + yoffset
                if xpos < 0 or xpos >= width:
                    continue
                if ypos < 0 or ypos >= height:
                    continue
                pixel_data[(xpos, ypos)] = c
    print("Write {}".format(filename))
    img.save(filename)


def extract_image_lidar_calib_by_frame(frame):
    (range_images, camera_projections, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame))

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                           range_images,
                                                           camera_projections,
                                                           range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    ts = frame.timestamp_micros
    images = sorted(frame.images, key=lambda i:i.name)

    # Only save FRONT camera image.
    frame_image = images[0]
    mask = tf.equal(cp_points_all_tensor[..., 0], frame_image.name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    camera_name = open_dataset.CameraName.Name.Name(frame_image.name)
    if not os.path.isdir(camera_name):
        os.makedirs(camera_name)
    filename = "{}/{}_calib.jpg".format(camera_name, ts)
    plot_points_on_image(projected_points_all_from_raw_data,
                         frame_image, filename, color_func)


def extract_image_lidar_calib(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        extract_image_lidar_calib_by_frame(frame)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord-file", "-f", help="Input .tfrecord file", required=True)
    args = parser.parse_args()
    extract_image_lidar_calib(args.tfrecord_file)

if __name__ == "__main__":
    main()
