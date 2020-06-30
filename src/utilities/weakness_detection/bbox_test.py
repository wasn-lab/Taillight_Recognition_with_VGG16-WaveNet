#!/usr/bin/env python
import unittest
from bbox import BBox, calc_iou


class BBoxTest(unittest.TestCase):
    def test_1(self):
        box1 = BBox()
        box1.left_x, box1.right_x = 10, 20
        box1.top_y, box1.bottom_y = 12, 30
        box2 = BBox()
        box2.left_x, box2.right_x = 13, 27
        box2.top_y, box2.bottom_y = 15, 20
        self.assertTrue(calc_iou(box1, box2) - 35 / 315.0 < 0.01)

        box2.left_x, box2.right_x = 25, 30
        box2.top_y, box2.bottom_y = 15, 20
        self.assertTrue(calc_iou(box1, box2) == 0)

if __name__ == "__main__":
    unittest.main()
