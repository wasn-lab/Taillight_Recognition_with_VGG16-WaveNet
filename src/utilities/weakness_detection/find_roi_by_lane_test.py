import unittest
import os
import cv2

from find_roi_by_lane import find_roi_by_lane_instance, fill_roi


class ROILaneTest(unittest.TestCase):
    def setUp(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))

    def test_2(self):
        img_filename = os.path.join(self.cur_dir, "test", "lane_instance_2.png")
        roi = find_roi_by_lane_instance(img_filename)
        self.assertEqual(roi, [(247, 117), (275, 118), (468, 255), (54, 255)])
        # filled = fill_roi(img_filename, roi)
        # cv2.imwrite("1.png", filled)

    def test_3(self):
        img_filename = os.path.join(self.cur_dir, "test", "lane_instance_3.png")
        roi = find_roi_by_lane_instance(img_filename)
        self.assertEqual(roi, [(222, 98), (250, 95), (277, 101), (466, 194), (466, 255), (11, 255), (0, 167)])
        # filled = fill_roi(img_filename, roi)
        # cv2.imwrite("3.png", filled)

if __name__ == "__main__":
    unittest.main()
