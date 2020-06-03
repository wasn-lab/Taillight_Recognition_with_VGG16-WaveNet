#!/usr/bin/env python
from image_consts import RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT


class YoloBBox(object):
    def __init__(self, yolo_object):
        """
        (0, 0) is the left-top corner of image.
        """
        relative_coordinates = yolo_object["relative_coordinates"]
        _cx = relative_coordinates["center_x"] * RAW_IMAGE_WIDTH
        _cy = relative_coordinates["center_y"] * RAW_IMAGE_HEIGHT
        bwidth = relative_coordinates["width"] * RAW_IMAGE_WIDTH
        bheight = relative_coordinates["height"] * RAW_IMAGE_HEIGHT
        left_x = max(0, _cx - bwidth / 2)
        top_y = max(0, _cy - bheight / 2)
        right_x = min(RAW_IMAGE_WIDTH - 1, _cx + bwidth / 2)
        bottom_y = min(RAW_IMAGE_HEIGHT - 1, _cy + bheight / 2)
        self.left_x = int(left_x)
        self.top_y = int(top_y)
        self.right_x = int(right_x)
        self.bottom_y = int(bottom_y)
        self.name = yolo_object["name"]
        self.class_id = yolo_object["class_id"]
        self.confidence = yolo_object["confidence"]

    def as_tuple(self):
        """return (left_x, top_y, right_x, bottom_y)"""
        return (self.left_x, self.top_y, self.right_x, self.bottom_y)

    def is_on_border(self, x, y):
        """Return True if (x,y) is on bbox border."""
        left_x, top_y, right_x, bottom_y = self.as_tuple()
        if x == left_x or x == right_x:
            return bool(y >= top_y and y <= bottom_y)
        if y == top_y or y == bottom_y:
            return bool(x >= left_x and x <= right_x)
        return False

    def is_within(self, x, y):
        return bool(x >= self.left_x and x <= self.right_x and y >= self.top_y and y <= self.bottom_y)
