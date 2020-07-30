#!/usr/bin/env python
import unittest
import os
from efficientdet_mgr import EfficientDetMgr
from nn_labels import EfficientDetLabel

class EfficientDetMgrTest(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.json_file = os.path.join(cur_dir, "test", "efficientdet.json")

    def test_1(self):
        mgr = EfficientDetMgr(self.json_file)
        self.assertEqual(len(mgr.bboxes), 10)
        bbox = mgr.bboxes[0]
        self.assertEqual(bbox.left_x, 344)
        self.assertEqual(bbox.top_y, 202)
        self.assertEqual(bbox.right_x, 434)
        self.assertEqual(bbox.bottom_y, 289)
        self.assertEqual(bbox.name, "bus")
        self.assertEqual(bbox.class_id, EfficientDetLabel.BUS)


if __name__ == "__main__":
    unittest.main()
