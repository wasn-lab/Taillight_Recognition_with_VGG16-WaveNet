import unittest
from iou_utils import calc_iou, calc_iou5

class IOUUtilsTest(unittest.TestCase):
    def test_calc_iou(self):
        # left_x, top_y, right_x, bottom_y
        box1 = [1458, 208, 1919, 593]
        box2 = [1458, 210, 1918, 580]
        iou = calc_iou(box1, box2)
        self.assertTrue(iou > 0.95)

        box1 = [10, 12, 20, 30]
        box2 = [13, 15, 27, 20]
        self.assertTrue(calc_iou(box1, box2) - 35 / 180.0 < 0.01)

        box1 = [10, 12, 20, 30]
        box2 = [25, 15, 30, 20]
        self.assertTrue(calc_iou(box1, box2) == 0)

    def test_calc_iou5(self):
        # class_id, left_x, top_y, right_x, bottom_y
        box1 = [2, 1458, 208, 1919, 593]
        box2 = [2, 1458, 210, 1918, 580]
        iou = calc_iou5(box1, box2)
        self.assertTrue(iou > 0.95)

if __name__ == "__main__":
    unittest.main()
