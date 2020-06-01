#!/usr/bin/env python
import os
import unittest
from yolo_mgr import YoloMgr
from yolo_bbox import YoloBBox


TEST_DIR, _ = os.path.split(os.path.abspath(__file__))

class YoloBBoxTest(unittest.TestCase):
    def setUp(self):
        result_json = os.path.join(TEST_DIR, "test", "yolo_result.json")
        self.mgr = YoloMgr(result_json)

    def test_bbox_coord(self):
        obj = self.mgr.frames[0]["objects"][0]
        bbox = YoloBBox(obj)
        coord = obj["relative_coordinates"]

        self.assertEqual(coord["width"], 0.043834)
        self.assertEqual(coord["center_x"], 0.194974)
        self.assertEqual(coord["height"], 0.055284)
        self.assertEqual(coord["center_y"], 0.278952)

        self.assertEqual(bbox.as_tuple(), (105, 96, 131, 117))

        self.assertTrue(bbox.is_on_border(105, 100))
        self.assertTrue(bbox.is_on_border(105, 96))
        self.assertTrue(bbox.is_on_border(130, 117))
        self.assertFalse(bbox.is_on_border(105, 118))
        self.assertFalse(bbox.is_on_border(0, 100))


if __name__ == "__main__":
    unittest.main()
