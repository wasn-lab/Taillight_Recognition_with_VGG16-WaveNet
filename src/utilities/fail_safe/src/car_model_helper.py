#!/usr/bin/env python
import io
import os


def get_car_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    car_model_text = os.path.join(
        cur_dir, "..", "..", "..", "..",
        "build", "car_model", "scripts", "car_model.txt")
    with io.open(car_model_text) as _fp:
        car_model = _fp.read()
    return car_model
