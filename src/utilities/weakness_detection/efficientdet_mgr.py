#!/usr/bin/env python
import os
import logging


class EfficientDetMgr(object):
    def __init__(self, json_file):
        self.labels = self.__read_efficientdet_output(json_file)

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
