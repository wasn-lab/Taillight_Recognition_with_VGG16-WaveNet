#!/usr/bin/env python
import os
import unittest
from yolo_utils import get_bbox_coord, read_result_json, is_on_bbox_border


TEST_DIR, _ = os.path.split(os.path.abspath(__file__))

class YoloUtilsTest(unittest.TestCase):
    def setUp(self):
        result_json = os.path.join(TEST_DIR, "test", "yolo_result.json")
        self.bboxes = read_result_json(result_json)

    def test_bbox_coord(self):
        coord = get_bbox_coord(self.bboxes[0]["objects"][0]["relative_coordinates"])
        self.assertEqual(coord, (105, 96, 131, 117))

    def test_is_on_bbox_border(self):
        bbox = (105, 96, 131, 117)
        self.assertTrue(is_on_bbox_border(bbox, 105, 100))
        self.assertTrue(is_on_bbox_border(bbox, 105, 96))
        self.assertTrue(is_on_bbox_border(bbox, 130, 117))
        self.assertFalse(is_on_bbox_border(bbox, 105, 118))
        self.assertFalse(is_on_bbox_border(bbox, 0, 100))


if __name__ == "__main__":
    unittest.main()
