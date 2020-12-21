#!/usr/bin/env python
import io
import os
import logging
import configparser

def _get_ini_filename(car_model=None):
    if car_model is None:
        car_model = get_car_model()
    inis = {"B1_V2": "sb_b1.ini",
            "B1_V3": "sb_b1.ini",
            "C1": "sb_c1.ini"}

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ini_file = os.path.join(cur_dir, inis[car_model])
    if not os.path.isfile(ini_file):
        logging.error("Cannot find ini file: %s", ini_file)
    return ini_file


def get_sb_config(car_model=None):
    cfg = configparser.ConfigParser()
    cfg.read(_get_ini_filename(car_model))
    return {key: cfg["south_bridge"][key] for key in cfg["south_bridge"]}


def get_car_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    car_model_text = os.path.join(
        cur_dir, "..", "..", "..",
        "build", "car_model", "scripts", "car_model.txt")
    with io.open(car_model_text) as _fp:
        car_model = _fp.read()
    return car_model
