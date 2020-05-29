#!/usr/bin/env python
import json
import os
import io

IMAGE_WIDTH = 608
IMAGE_HEIGHT = 384

def get_bbox_coord(relative_coordinates):
    """return (left_x, top_y, right_x, bottom_y)"""
    cx = relative_coordinates["center_x"] * IMAGE_WIDTH
    cy = relative_coordinates["center_y"] * IMAGE_HEIGHT
    bwidth = relative_coordinates["width"] * IMAGE_WIDTH
    bheight = relative_coordinates["height"] * IMAGE_HEIGHT
    left_x = max(0, cx - bwidth / 2)
    top_y = max(0, cy - bheight / 2)
    right_x = min(IMAGE_WIDTH - 1, cx + bwidth / 2)
    bottom_y = min(IMAGE_HEIGHT - 1, cy + bheight / 2)
    return (int(left_x), int(top_y), int(right_x), int(bottom_y))


def read_result_json(filename):
    if not os.path.isfile(filename):
        logging.error("File not found: %s", filename)
        return []
    with io.open(filename, encoding="utf-8") as _fp:
        jdata = json.load(_fp)
    return jdata


def is_on_bbox_border(bbox, x, y):
    """Return True if (x,y) is on bbox border."""
    left_x, top_y, right_x, bottom_y = bbox
    if x == left_x or x == right_x:
        return bool(y >= top_y and y <= bottom_y)
    if y == top_y or y == bottom_y:
        return bool(x >= left_x and x <= right_x)
    return False
