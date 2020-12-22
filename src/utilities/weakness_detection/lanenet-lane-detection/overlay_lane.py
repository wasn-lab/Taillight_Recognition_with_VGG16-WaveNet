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
import logging
import os
import multiprocessing
import cv2
import numpy as np
from find_lanes import image_paths

COLORS = {1: (0, 255, 0),
          2: (255, 0, 0),
          3: (0, 0, 255),
          4: (127, 127, 0)}

def overlay_lanes(image_filename):
    logging.warning("Process %s", image_filename)
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    lane_filename = image_filename[:-4] + "_lane_instance_8uc1.png"
    if not os.path.isfile(lane_filename):
        logging.error("File not found: %s", lane_filename)
        return

    lane_image = cv2.imread(lane_filename, cv2.IMREAD_UNCHANGED)
    output_image = np.zeros_like(image)

    scale_r = float(lane_image.shape[0]) / image.shape[0]
    scale_c = float(lane_image.shape[1]) / image.shape[1]
    for r in range(image.shape[0]):
        rl = int(r * scale_r)
        for c in range(image.shape[1]):
            cl = int(c * scale_c)
            if lane_image[rl][cl] == 0:
                output_image[r][c] = image[r][c]
            else:
                output_image[r][c] = COLORS[lane_image[rl][cl]]
    output_filename = image_filename[:-4] + "_lane_overlay.jpg"
    logging.warning("Write %s", output_filename)
    cv2.imwrite(output_filename, output_image)


def main():
    """Prog entry"""
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-filenames', required=True)
    args = parser.parse_args()
    image_filenames = image_paths(args.image_filenames)

    nproc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nproc)

    res = pool.map(overlay_lanes, image_filenames)

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
