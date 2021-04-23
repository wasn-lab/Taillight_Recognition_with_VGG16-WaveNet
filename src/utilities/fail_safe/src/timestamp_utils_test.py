# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
from timestamp_utils import get_timestamp_mot


class TimestampUtilsTest(unittest.TestCase):
    def test_1(self):
        ret = get_timestamp_mot()
        self.assertEqual(len(ret), 13)
        self.assertTrue(ret.isdigit())


if __name__ == "__main__":
    unittest.main()
