#!/usr/bin/env python
import os
import io
import json
import logging

def read_result_json(filename):
    if not os.path.isfile(filename):
        logging.error("File not found: %s", filename)
        return []
    with io.open(filename, encoding="utf-8") as _fp:
        jdata = json.load(_fp)
    return jdata


class YoloMgr(object):
    RAW_IMAGE_WIDTH = 608
    RAW_IMAGE_HEIGHT = 384
    def __init__(self, json_file):
        self.frames = read_result_json(json_file)
