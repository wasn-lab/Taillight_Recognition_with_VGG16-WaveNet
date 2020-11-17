#!/usr/bin/env python
import subprocess
from sb_rosbag_sender import get_sb_config


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
