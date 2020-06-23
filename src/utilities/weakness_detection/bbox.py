#!/usr/bin/env python


def calc_iou(box1, box2):
    if box1.class_id != box2.class_id:
        return 0
    left_x = max(box1.left_x, box2.left_x)
    right_x = min(box1.right_x, box2.right_x)
    top_y = max(box1.top_y, box2.top_y)
    bottom_y = min(box1.bottom_y, box2.bottom_y)
    if left_x >= right_x or top_y <= bottom_y:
        return 0
    area1 = (box1.right_x - box1.left_x) * (box1.bottom_y - box1.top_y)
    area2 = (box2.right_x - box2.left_x) * (box2.bottom_y - box2.top_y)
    intersection = (right_x - left_x) * (bottom_y - top_y)
    return float(intersection) / (area1 + area2 - intersection)


class BBox(object):
    def __init__(self):
        """
        (0, 0) is the left-top corner of image.
        """
        self.left_x = 0
        self.top_y = 0
        self.right_x = 0
        self.bottom_y = 0
        self.name = ""
        self.class_id = 0
        self.confidence = 0

    def as_tuple(self):
        """return (left_x, top_y, right_x, bottom_y)"""
        return (self.left_x, self.top_y, self.right_x, self.bottom_y)

    def is_within(self, x, y):
        return bool(x >= self.left_x and x <= self.right_x and y >= self.top_y and y <= self.bottom_y)

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

    def __str__(self):
        return "{}({}): ({}, {}, {}, {}), confidence: {}".format(self.name, self.class_id, self.left_x, self.top_y, self.right_x, self.bottom_y, self.confidence)
