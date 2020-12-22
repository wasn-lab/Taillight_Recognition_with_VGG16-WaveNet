#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from car_model_helper import get_car_model, get_sb_config

class CarModelHelperTest(unittest.TestCase):
    def test_get_car_model(self):
        car_model = get_car_model()
        self.assertTrue(car_model in ["B1_V2", "B1_V3", "C1"])

    def test_get_sb_config(self):
        sb_config = get_sb_config("B1_V2")
        self.assertEqual(sb_config["license_plate_number"], u"試0002")
        sb_config = get_sb_config("B1_V3")
        self.assertEqual(sb_config["license_plate_number"], u"試0002")
        sb_config = get_sb_config("C1")
        self.assertEqual(sb_config["license_plate_number"], u"MOREA")

if __name__ == "__main__":
    unittest.main()
