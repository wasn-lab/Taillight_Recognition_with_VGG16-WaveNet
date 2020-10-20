#!/usr/bin/env python
"""
Functions/parameters that defines dangerous/cautious situation.
"""


def in_3d_roi(center_x, center_y):
    # If object appear in front of the car within 40 meters
    # and at side of the car by 1.5 meters, it is considered worth watching
    return bool(center_x > 0 and center_x <= 40 and abs(center_y) <= 1.5)
