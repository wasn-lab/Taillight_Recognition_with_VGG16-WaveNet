#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Change LED text when driving mode (auto/manual) changes.
"""
from __future__ import print_function
import os
import subprocess
import sys
import rospy
from msgs.msg import Flag_Info

MANUAL_DRIVING = 1
AUTO_DRIVING = 2

def get_powerled_exe():
    for path in os.environ.get("CMAKE_PREFIX_PATH").split(":"):
        powerled = os.path.join(path, "lib", "powerled", "powerled")
        if os.path.isfile(powerled):
            return powerled
    return ""


def change_led_text(mode):
    """
    Return -- 0: success
              1: failure
    """
    if mode == AUTO_DRIVING:
        print("Change powerled text to auto driving mode.")
    elif mode == MANUAL_DRIVING:
        print("Change powerled text to manual driving mode.")
    else:
        print("Undefined mode: {}".format(mode))
        return 1
    powerled = get_powerled_exe()
    powerled_dir, _base_name = os.path.split(powerled)
    cmd = [powerled, str(mode)]
    try:
        subprocess.check_output(cmd, cwd=powerled_dir)
    except subprocess.CalledProcessError:
        print("Fail to run command: {}".format(" ".join(cmd)))
        return 1
    return 0

class LEDManager(object):
    def __init__(self):
        self.driving_mode = 0
        self.prev_mode = 0

    def _cb(self, msg):
        if int(msg.Dspace_Flag08) == 1:
            self.driving_mode = AUTO_DRIVING
        else:
            self.driving_mode = MANUAL_DRIVING

        if self.driving_mode != self.prev_mode:
            change_led_text(self.driving_mode)
        self.prev_mode = self.driving_mode

    def run(self):
        node_name = "LEDManagerNode"
        rospy.init_node(node_name)
        rospy.logwarn("Init %s", node_name)
        rospy.Subscriber("/Flag_Info02", Flag_Info, self._cb)

        rate = rospy.Rate(1)  # FPS: 1

        while not rospy.is_shutdown():
            rate.sleep()


def main():
    exe = get_powerled_exe()
    if not exe:
        print("Cannot find powerled executable!")
        sys.exit(1)
    mgr = LEDManager()
    mgr.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
