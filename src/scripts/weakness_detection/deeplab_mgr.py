#!/usr/bin/env python
import os
import logging
import cv2


class DeeplabMgr(object):
    IMAGE_WIDTH = 513
    IMAGE_HEIGHT = 513
    def __init__(self, png_file):
        self.labels = self.__read_labels_by_deeplab_output(png_file)

    def get_label_by_xy(self, labels, x, y):
        """Return the label at the specific location (x, y)"""
        assert(x >= 0 and x < DeeplabMgr.IMAGE_WIDTH)
        assert(y >= 0 and y < DeeplabMgr.IMAGE_HEIGHT)
        return self.labels[y][x]

    def __read_labels_by_deeplab_output(self, png_file):
        """Each pixel in |png_file| is labels. EX: 15 is person."""
        if not os.path.isfile(png_file):
            logging.error("File not exist: %s", png_file)
            return None
        img = cv2.imread(png_file, cv2.CV_8UC1)
        #    pixel_map = {}
        #    for row in range(IMAGE_HEIGHT):
        #        for col in range(IMAGE_WIDTH):
        #            label = img[row][col]
        #            if label > 0:
        #                pixel_map[label] = 1 + pixel_map.get(label, 0)
        #    print(pixel_map)
        return img
