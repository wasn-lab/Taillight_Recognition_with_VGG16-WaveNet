#!/usr/bin/env python
import os
import unittest
from deeplab_mgr import to_raw_image_pos, LETTERBOX_BORDER, RAW_TO_DEEPLAB_SCALE, MIN_Y, MAX_Y


class YoloBBoxTest(unittest.TestCase):
    def test_to_raw_image_pos(self):
        raw_x, raw_y = 100, 150
        yolo_x, yolo_y = raw_x, raw_y + LETTERBOX_BORDER
        deeplab_x, deeplab_y = yolo_x * RAW_TO_DEEPLAB_SCALE, yolo_y * RAW_TO_DEEPLAB_SCALE

        _x, _y = to_raw_image_pos(deeplab_x, deeplab_y)
        self.assertEqual(_x, raw_x)
        self.assertEqual(_y, raw_y)

        self.assertEqual(to_raw_image_pos(0, MIN_Y), (0, 0))
        self.assertEqual(to_raw_image_pos(512, MAX_Y), (607, 383))


if __name__ == "__main__":
    unittest.main()
