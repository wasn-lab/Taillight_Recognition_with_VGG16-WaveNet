#!/usr/bin/env python
import argparse
import io
import json
import logging
import os

import cv2


def _get_bboxes(image_filename):
    _gt = image_filename[:-4] + ".json"
    with io.open(_gt, encoding="utf-8") as _fp:
        contents = _fp.read()
    return json.loads(contents)


def _draw_bboxes(image_filename, bboxes):
    img = cv2.imread(image_filename)
    thickness = 2
    color_map = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (0, 255, 127),
        5: (255, 255, 0),
        7: (0, 100, 0),
    }


    for doc in bboxes:
        box = doc["coordinates"]
        color = color_map.get(doc["class_id"], 0)
        if not doc.get("negative", False):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

    return img


def draw_bboxes(args):
    image_filename = args.image_filename
    output_dir = os.path.normpath(args.output_dir)
    bboxes = _get_bboxes(image_filename)

    img = _draw_bboxes(image_filename, bboxes)

    _, base_filename = os.path.split(image_filename)
    output_filename = os.path.join(output_dir, base_filename[:-4] + "_expect.jpg")

    cv2.imwrite(output_filename, img)
    logging.warning("Write %s", output_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-filename", "-i", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    draw_bboxes(args)

if __name__ == "__main__":
    main()
