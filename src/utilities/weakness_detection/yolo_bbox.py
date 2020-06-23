#!/usr/bin/env python
from image_consts import RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT
from bbox import BBox


def gen_bbox_by_yolo_object(yolo_object):
    """
    (0, 0) is the left-top corner of image.
    """
    relative_coordinates = yolo_object["relative_coordinates"]
    _cx = relative_coordinates["center_x"] * RAW_IMAGE_WIDTH
    _cy = relative_coordinates["center_y"] * RAW_IMAGE_HEIGHT
    bwidth = relative_coordinates["width"] * RAW_IMAGE_WIDTH
    bheight = relative_coordinates["height"] * RAW_IMAGE_HEIGHT
    box = BBox()
    left_x = max(0, _cx - bwidth / 2)
    top_y = max(0, _cy - bheight / 2)
    right_x = min(RAW_IMAGE_WIDTH - 1, _cx + bwidth / 2)
    bottom_y = min(RAW_IMAGE_HEIGHT - 1, _cy + bheight / 2)
    box.left_x = int(left_x)
    box.top_y = int(top_y)
    box.right_x = int(right_x)
    box.bottom_y = int(bottom_y)
    box.name = yolo_object["name"]
    box.class_id = yolo_object["class_id"]
    box.confidence = yolo_object["confidence"]
    return box
