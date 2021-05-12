#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import io
import os
import rospy


def get_car_model_as_str():
    if rospy.has_param("/car_model"):
        return rospy.get_param("/car_model")
    return "ITRIADV-DEFAULT"
