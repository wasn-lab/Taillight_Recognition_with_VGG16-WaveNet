#!/usr/bin/python
import os
import logging
import argparse
import cv2


def pick_if_necessary(img_path):
    logging.debug("Handle %s", img_path)
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    if height < 1000 or width < 1000:
        logging.debug("Skip: %s with size (%d %d)", img_path, width, height)
        return
    txt_path = img_path[:-4] + ".txt"
    if not os.path.isfile(txt_path):
        logging.debug("Skip: %s not found", txt_path)
        return
    print(img_path)


def pick_images(image_path):
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                fullpath = os.path.join(root, filename)
                pick_if_necessary(fullpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    args = parser.parse_args()
    pick_images(args.image_path)


if __name__ == "__main__":
    main()
