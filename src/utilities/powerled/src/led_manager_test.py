# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import os
from led_manager import (get_powerled_exe, change_led_text, LEDManager,
    AUTO_DRIVING, MANUAL_DRIVING)

class LEDManagerTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_powerled_exe(self):
        exe = get_powerled_exe()
        self.assertTrue(exe.endswith("powerled"))
        self.assertTrue(os.access(exe, os.X_OK))

    def test_change_led_text(self):
        ret = change_led_text(AUTO_DRIVING)
        self.assertEqual(ret, 0)
        ret = change_led_text(MANUAL_DRIVING)
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
