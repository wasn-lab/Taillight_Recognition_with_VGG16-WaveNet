#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import io
import os
import rospy


def get_car_model_as_str():
    if rospy.has_param("/car_model"):
        return rospy.get_param("/car_model")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    car_model_text = os.path.join(
        cur_dir, "..", "..", "..", "..",
        "build", "car_model", "scripts", "car_model.txt")
    with io.open(car_model_text) as _fp:
        car_model = _fp.read()
    return car_model
