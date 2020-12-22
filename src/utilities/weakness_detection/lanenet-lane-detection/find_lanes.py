#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import time
import io
import logging

import cv2
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-filenames', type=str)
    parser.add_argument('--weights-path', type=str)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def image_paths(image_list_path):
    with io.open(image_list_path) as _fp:
        contents = _fp.read()
    return [_.strip() for _ in contents.splitlines()]


def instanciate(mask_image):
    instance_image_8uc1 = np.zeros(
        (mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
    instances = {(0, 0, 0) : 0}
    for r in range(mask_image.shape[0]):
        for c in range(mask_image.shape[1]):
            color = tuple(mask_image[r][c])
            if color not in instances:
                instances[color] = len(instances)
            instance_image_8uc1[r][c] = instances[color]
    return instance_image_8uc1

def test_lanenet(image_filenames, weights_path):
    """
    :param image_path:
    :param weights_path:
    :return:
    """
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    t_start = time.time()
    num_images = 0
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        for image_filename in image_paths(image_filenames):
            logging.warning("Process %s", image_filename)
            num_images += 1
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]})

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis)
            mask_image = postprocess_result['mask_image']

            lane_filename = image_filename[:-4] + "_lane_instance.png"
            if mask_image is not None:
                cv2.imwrite(lane_filename, mask_image)
                logging.warning("Write %s", lane_filename)
                # Not needed as we can use mask_image to find ROI.
                # lane_filename_8uc1 = image_filename[:-4] + "_lane_instance_8uc1.png"
                # instance_8uc1 = instanciate(mask_image)
                # cv2.imwrite(lane_filename_8uc1, instance_8uc1)
                # logging.warning("Write %s", lane_filename_8uc1)
            else:
                logging.warning("Cannot write %s", lane_filename)

    sess.close()
    t_elapse = time.time() - t_start
    logging.warning("Total time: %s seconds, avg per image: %s seconds",
                    t_elapse, t_elapse / num_images)

    return


def main():
    """Prog entry"""
    args = init_args()
    test_lanenet(args.image_filenames, args.weights_path)

if __name__ == '__main__':
    main()
