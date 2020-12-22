#!/usr/bin/env python
import os
import logging
import cv2
from image_consts import (
    RAW_IMAGE_WIDTH,
    RAW_IMAGE_HEIGHT,
    DEEPLAB_IMAGE_WIDTH,
    DEEPLAB_IMAGE_HEIGHT)


def deeplab_pos_to_raw_pos(x, y, scale=0):
    """
    Raw image: 608x384, deeplab: 513x513.
    To get deeplab input: 608x384 -> 608x608 (letterbox) -> 513x513
    """
    if scale == 0:
        scale = float(DEEPLAB_IMAGE_WIDTH) / RAW_IMAGE_WIDTH
    _x = int(x / scale + 0.5)
    _y = int(y / scale + 0.5) - 112
    return (_x, _y)


def raw_image_pos_to_deeplab_pos(raw_x, raw_y, org_width=RAW_IMAGE_WIDTH, org_height=RAW_IMAGE_HEIGHT):
    """
    Map the raw image position (raw_x, raw_y) to the corresponding
    deeplab position.
    """
    letterbox_border_height = (org_width - org_height) / 2
    scale = float(DEEPLAB_IMAGE_WIDTH) / org_width
    _x = int(raw_x * scale + 0.5);
    _x = min(_x, DEEPLAB_IMAGE_WIDTH - 1)
    _y = int((raw_y + letterbox_border_height) * scale + 0.5)
    _y = min(_y, DEEPLAB_IMAGE_HEIGHT - 1)
    return (_x, _y)


def get_deeplab_min_y(org_width=RAW_IMAGE_WIDTH, org_height=RAW_IMAGE_HEIGHT):
    border = (org_width - org_height) / 2
    scale = DEEPLAB_IMAGE_WIDTH / float(org_width)
    return int(border * scale + 0.5)

def get_deeplab_max_y(org_width=RAW_IMAGE_WIDTH, org_height=RAW_IMAGE_HEIGHT):
    border = (org_width - org_height) / 2
    scale = DEEPLAB_IMAGE_WIDTH / float(org_width)
    return int((border + org_height - 1) * scale + 0.5)


class DeeplabMgr(object):
    def __init__(self, png_file):
        self.labels = self.__read_labels_by_deeplab_output(png_file)

    def get_label_by_xy(self, x, y):
        """Return the label at the specific location (x, y)"""
        return self.labels[y][x]

    def __read_labels_by_deeplab_output(self, png_file):
        """Each pixel in |png_file| is labels. EX: 15 is person."""
        if not os.path.isfile(png_file):
            logging.error("File not exist: %s", png_file)
            return None
        img = cv2.imread(png_file, cv2.CV_8UC1)
        return img

    def count_labels(self):
        pixel_map = {}
        for row in range(DEEPLAB_IMAGE_HEIGHT):
            for col in range(DEEPLAB_IMAGE_WIDTH):
                label = self.labels[row][col]
                if label > 0:
                    pixel_map[label] = 1 + pixel_map.get(label, 0)
        return pixel_map
