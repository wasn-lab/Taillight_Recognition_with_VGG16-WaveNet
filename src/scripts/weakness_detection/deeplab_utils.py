#!/usr/bin/env python
import os
import argparse
import logging
import cv2

IMAGE_WIDTH = 513
IMAGE_HEIGHT = 513


def read_labels_by_deeplab_output(png_file):
    """Each pixel in |png_file| is labels. EX: 15 is person."""
    if not os.path.isfile(png_file):
        logging.error("File not exist: %s", png_file)
        return None
    img = cv2.imread(png_file, cv2.CV_8UC1)
    pixel_map = {}
    assert(img.shape[0] == IMAGE_HEIGHT)
    assert(img.shape[1] == IMAGE_WIDTH)
#    for row in range(IMAGE_HEIGHT):
#        for col in range(IMAGE_WIDTH):
#            label = img[row][col]
#            if label > 0:
#                pixel_map[label] = 1 + pixel_map.get(label, 0)
#    print(pixel_map)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-file", "-f", help="labels marked by deeplab", required=True)
    args = parser.parse_args()
    read_labels_by_deeplab_output(args.png_file)


if __name__ == "__main__":
    main()
