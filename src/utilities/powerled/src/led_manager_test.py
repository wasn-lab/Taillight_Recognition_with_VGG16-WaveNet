# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import os
from led_manager import (get_powerled_exe, change_led_text, LEDManager,
    AUTO_DRIVING, MANUAL_DRIVING)
from std_msgs.msg import Bool

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

    def test_cb(self):
        led_mgr = LEDManager()

        self.assertEqual(led_mgr._cb(Bool(True)), 1)
        # Don't change text when self-driving state does not change.
        self.assertEqual(led_mgr._cb(Bool(True)), 0)
        self.assertEqual(led_mgr._cb(Bool(True)), 0)

        self.assertEqual(led_mgr._cb(Bool(False)), 1)
        self.assertEqual(led_mgr._cb(Bool(False)), 0)
        self.assertEqual(led_mgr._cb(Bool(True)), 1)

if __name__ == "__main__":
    unittest.main()
