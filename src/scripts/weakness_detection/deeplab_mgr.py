#!/usr/bin/env python
import os
import logging
import cv2
from image_consts import (
    LETTERBOX_BORDER_WIDTH,
    DEEPLAB_RAW_TO_DEEPLAB_SCALE,
    DEEPLAB_IMAGE_WIDTH,
    DEEPLAB_IMAGE_HEIGHT)


def deeplab_pos_to_raw_pos(x, y):
    """
    Raw image: 608x384, deeplab: 513x513.
    To get deeplab input: 608x384 -> 608x608 (letterbox) -> 513x513
    """
    _x = int(x / DEEPLAB_RAW_TO_DEEPLAB_SCALE + 0.5)
    _y = int(y / DEEPLAB_RAW_TO_DEEPLAB_SCALE + 0.5) - 112
    return (_x, _y)


def raw_image_pos_to_deeplab_pos(raw_x, raw_y):
    """
    Map the raw image position (raw_x, raw_y) to the corresponding
    deeplab position.
    """
    yolo_x, yolo_y = raw_x, raw_y + LETTERBOX_BORDER_WIDTH
    deeplab_x, deeplab_y = yolo_x * DEEPLAB_RAW_TO_DEEPLAB_SCALE, yolo_y * DEEPLAB_RAW_TO_DEEPLAB_SCALE

    return (int(deeplab_x + 0.5), int(deeplab_y + 0.5))


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
