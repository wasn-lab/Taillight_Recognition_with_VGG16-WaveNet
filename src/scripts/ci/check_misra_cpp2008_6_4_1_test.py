#!/usr/bin/env python
import unittest
from check_misra_cpp2008_6_4_1 import check_misra_cpp2008_6_4_1_by_cpp


class MISRA6_4_1Test(unittest.TestCase):
    def test_1(self):
        cpp = "src/sensing/itri_drivenet/drivenet/src/drivenet_120_1_b1.cpp"
        check_misra_cpp2008_6_4_1_by_cpp(cpp)


if __name__ == "__main__":
    unittest.main()
