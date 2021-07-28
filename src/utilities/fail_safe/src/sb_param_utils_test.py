# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import rospy
from sb_param_utils import get_vid, get_license_plate_number


class SBParamUtilsTest(unittest.TestCase):
    def test_get_vid(self):
        param_name = "/south_bridge/vid"
        if rospy.has_param(param_name):
            org_vid = rospy.get_param(param_name)
        else:
            org_vid = None
        rospy.set_param("/south_bridge/vid", "0c9d92107992")
        vid = get_vid()
        self.assertEqual(vid, "0c9d92107992")

        rospy.delete_param(param_name)
        vid = get_vid()
        self.assertEqual(vid, "unknown_vid")

        if org_vid:
            rospy.set_param(param_name, org_vid)

    def test_get_license_plate_number(self):
        param_name = "/south_bridge/license_plate_number"
        if rospy.has_param(param_name):
            org_val = rospy.get_param(param_name)
        else:
            org_val = None
        rospy.set_param(param_name, "0c9d92107992")
        actual = get_license_plate_number()
        self.assertEqual(actual, "0c9d92107992")

        rospy.delete_param(param_name)
        actual = get_license_plate_number()
        self.assertEqual(actual, u"unknown_license_plate_number")

        if org_val:
            rospy.set_param(param_name, org_val)


if __name__ == "__main__":
    unittest.main()
