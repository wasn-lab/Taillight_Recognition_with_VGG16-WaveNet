#!/usr/bin/env python
import subprocess
import io
import os
from sb_rosbag_sender import get_sb_config
from car_model_helper import get_car_model


def main():
    cfg = get_sb_config()
    for key in ["vid", "license_plate_number", "company_name"]:
        param = "/south_bridge/{}".format(key)
        cmd = ["rosparam", "set", param, cfg[key]]
        print(" ".join(cmd))
        output = subprocess.check_output(cmd)
        print(output)
    cmd = ["rosparam", "set", "car_model", get_car_model()]
    print(" ".join(cmd))
    output = subprocess.check_output(cmd)
    print(output)


if __name__ == "__main__":
    main()
