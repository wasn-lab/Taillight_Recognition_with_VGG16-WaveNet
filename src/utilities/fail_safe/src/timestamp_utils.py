# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import time

def get_timestamp_mot():
    """
    Return the current timestamp in 13-digit string.
    The 13-digit format is demanded by MOT.'
    """
    return int(time.time() * 1000)
