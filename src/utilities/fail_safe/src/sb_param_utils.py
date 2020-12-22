# -*- encoding: utf-8 -*-
import configparser
import os
import rospy
import rospkg
from car_model_helper import get_car_model


def get_vid():
    return __get_sb_param("vid")


def get_license_plate_number():
    return __get_sb_param("license_plate_number")


def __get_sb_param(sb_param_name):
    key = "/south_bridge/{}".format(sb_param_name)
    if rospy.has_param(key):
        return rospy.get_param(key)
    # fall-back
    inis = {"B1_V2": "sb_b1.ini",
            "B1_V3": "sb_b1.ini",
            "C1": "sb_c1.ini"}
    car_model = get_car_model()
    car_model_dir = rospkg.RosPack().get_path("car_model")
    sb_ini = os.path.join(car_model_dir, "south_bridge", inis[car_model])
    cfg = configparser.ConfigParser()
    cfg.read(sb_ini)
    return cfg["south_bridge"][sb_param_name]
