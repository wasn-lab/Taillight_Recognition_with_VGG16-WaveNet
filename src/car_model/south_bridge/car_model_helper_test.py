#!/usr/bin/env python
import unittest
from car_model_helper import get_car_model

class CarModelHelperTest(unittest.TestCase):
    def test_get_car_model(self):
        car_model = get_car_model()
        self.assertTrue(car_model in ["B1_V2", "B1_V3", "C1"])


if __name__ == "__main__":
    unittest.main()
