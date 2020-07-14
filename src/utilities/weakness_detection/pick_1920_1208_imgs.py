#!/usr/bin/python
import os
import shutil
import logging
import argparse
import cv2


def cp_if_necessary(img_path, output_path):
    logging.warning("Handle %s", img_path)
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    if height < 1000 or width < 1000:
        logging.warning("Skip: %s with size (%d %d)", img_path, width, height)
        return
    txt_path = img_path[:-4] + ".txt"
    if not os.path.isfile(txt_path):
        logging.warning("Skip: %s not found", txt_path)
        return
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    logging.warning("Find %s", img_path)
    shutil.copy(img_path, output_path)
    shutil.copy(txt_path, output_path)


def cp_images(image_path, output_path):
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                fullpath = os.path.join(root, filename)
                cp_if_necessary(fullpath, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", default="out")
    args = parser.parse_args()
    cp_images(args.image_path, args.output_path)


if __name__ == "__main__":
    main()
