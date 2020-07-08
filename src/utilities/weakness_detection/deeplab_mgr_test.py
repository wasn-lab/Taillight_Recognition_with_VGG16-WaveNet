#!/usr/bin/env python
import unittest
from deeplab_mgr import deeplab_pos_to_raw_pos, raw_image_pos_to_deeplab_pos
from image_consts import (
    DEEPLAB_MIN_Y, DEEPLAB_MAX_Y, DEEPLAB_IMAGE_WIDTH, RAW_IMAGE_WIDTH,
    RAW_IMAGE_HEIGHT)


class DeeplabMgrTest(unittest.TestCase):
    def test_deeplab_pos_to_raw_pos(self):
        raw_x, raw_y = 100, 150
        deeplab_x, deeplab_y = raw_image_pos_to_deeplab_pos(raw_x, raw_y)

        _x, _y = deeplab_pos_to_raw_pos(deeplab_x, deeplab_y)
        self.assertEqual(_x, raw_x)
        self.assertEqual(_y, raw_y)

        raw_x, raw_y = deeplab_pos_to_raw_pos(0, DEEPLAB_MIN_Y)
        self.assertEqual(raw_x, 0)
        self.assertTrue(raw_y <= 1)  # rounding error

        self.assertEqual(
            deeplab_pos_to_raw_pos(DEEPLAB_IMAGE_WIDTH - 1, DEEPLAB_MAX_Y),
            (RAW_IMAGE_WIDTH - 1, RAW_IMAGE_HEIGHT - 1))


if __name__ == "__main__":
    unittest.main()
