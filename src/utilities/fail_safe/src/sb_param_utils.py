# -*- encoding: utf-8 -*-
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import configparser
import os
import rospy
import rospkg
from car_model_helper import get_car_model_as_str


def get_vid():
    return __get_sb_param("vid")


def get_license_plate_number():
    return __get_sb_param("license_plate_number")


def get_company_name():
    return __get_sb_param("company_name")


def __get_sb_param(sb_param_name):
    key = "/south_bridge/{}".format(sb_param_name)
    if rospy.has_param(key):
        return rospy.get_param(key)
    return "unknown_" + sb_param_name
