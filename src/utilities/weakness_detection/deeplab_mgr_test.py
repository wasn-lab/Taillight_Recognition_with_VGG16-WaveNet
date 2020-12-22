#!/usr/bin/env python
import unittest
from deeplab_mgr import deeplab_pos_to_raw_pos, raw_image_pos_to_deeplab_pos
from image_consts import (DEEPLAB_IMAGE_WIDTH, RAW_IMAGE_WIDTH,
                          RAW_IMAGE_HEIGHT)


class DeeplabMgrTest(unittest.TestCase):
    def test_deeplab_pos_to_raw_pos(self):
        raw_x, raw_y = 100, 150
        deeplab_x, deeplab_y = raw_image_pos_to_deeplab_pos(raw_x, raw_y)

        _x, _y = deeplab_pos_to_raw_pos(deeplab_x, deeplab_y)
        self.assertEqual(_x, raw_x)
        self.assertEqual(_y, raw_y)

        raw_x, raw_y = 817, 360
        x, y = raw_image_pos_to_deeplab_pos(raw_x, raw_y, 1280, 720)
        self.assertTrue(abs(x - 327) <= 1)
        self.assertTrue(abs(y - 256) <= 1)

if __name__ == "__main__":
    unittest.main()
