#!/usr/bin/env python
import subprocess
import io
import os
from sb_rosbag_sender import get_sb_config


def get_car_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    car_model_text = os.path.join(
        cur_dir, "..", "..", "..",
        "build", "car_model", "scripts", "car_model.txt")
    with io.open(car_model_text) as _fp:
        car_model = _fp.read()
    return car_model


def main():
    cfg = get_sb_config()
    for key in ["vid", "license_plate_number", "company_name"]:
        param = "/south_bridge/{}".format(key)
        cmd = ["rosparam", "set", param, cfg[key]]
        print(" ".join(cmd))
        output = subprocess.check_output(cmd)
        print(output)


if __name__ == "__main__":
    main()
