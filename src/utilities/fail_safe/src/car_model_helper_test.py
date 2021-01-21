# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
from car_model_helper import get_car_model_as_str


class CarModelTest(unittest.TestCase):
    def setUp(self):
        if rospy.has_param("/car_model"):
            self.org_car_model = rospy.get_param("/car_model")
        else:
            self.org_car_model = None

    def tearDown(self):
        if self.org_car_model:
            rospy.set_param("/car_model", self.org_car_model)
        else:
            rospy.delete_param("/car_model")

    def test_get_car_model_as_str(self):
        car_model = get_car_model_as_str()
        self.assertTrue(len(car_model) > 0)

        rospy.set_param("/car_model", "B1_V2")
        self.assertEqual(get_car_model_as_str(), "B1_V2")
        rospy.set_param("/car_model", "B1_V3")
        self.assertEqual(get_car_model_as_str(), "B1_V3")
        rospy.set_param("/car_model", "C1")
        self.assertEqual(get_car_model_as_str(), "C1")

if __name__ == "__main__":
    unittest.main()
