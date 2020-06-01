#!/usr/bin/env python
from yolo_mgr import YoloMgr


class YoloBBox(object):
    def __init__(self, relative_coordinates):
        _cx = relative_coordinates["center_x"] * YoloMgr.RAW_IMAGE_WIDTH
        _cy = relative_coordinates["center_y"] * YoloMgr.RAW_IMAGE_HEIGHT
        bwidth = relative_coordinates["width"] * YoloMgr.RAW_IMAGE_WIDTH
        bheight = relative_coordinates["height"] * YoloMgr.RAW_IMAGE_HEIGHT
        left_x = max(0, _cx - bwidth / 2)
        top_y = max(0, _cy - bheight / 2)
        right_x = min(YoloMgr.RAW_IMAGE_WIDTH - 1, _cx + bwidth / 2)
        bottom_y = min(YoloMgr.RAW_IMAGE_HEIGHT - 1, _cy + bheight / 2)
        self.left_x = int(left_x)
        self.top_y = int(top_y)
        self.right_x = int(right_x)
        self.bottom_y = int(bottom_y)

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
