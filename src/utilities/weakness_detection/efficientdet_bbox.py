#!/usr/bin/env python
from image_consts import RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT


class EfficientBBox(object):
    def __init__(self):
        """
        (0, 0) is the left-top corner of image.
        """
        self.left_x = 0
        self.top_y = 0
        self.right_x = 0
        self.bottom_y = 0
        self.class_id = -1
        self.score = 0

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
