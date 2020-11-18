import configparser
import os
import rospy
import rospkg


def get_vid():
    return __get_sb_param("vid")


def get_license_plate_number():
    return __get_sb_param("license_plate_number")


def __get_sb_param(sb_param_name):
    key = "/south_bridge/{}".format(sb_param_name)
    if rospy.has_param(key):
        return rospy.get_param(key)
    car_model_dir = rospkg.RosPack().get_path("car_model")
    sb_ini = os.path.join(car_model_dir, "south_bridge", "sb.ini")
    cfg = configparser.ConfigParser()
    cfg.read(sb_ini)
    return cfg["south_bridge"][sb_param_name]
