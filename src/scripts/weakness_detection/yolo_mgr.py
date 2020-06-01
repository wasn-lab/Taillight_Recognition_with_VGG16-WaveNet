#!/usr/bin/env python
import os
import io
import json
import logging
from deeplab_mgr import DeeplabMgr, to_raw_image_pos
from image_consts import DEEPLAB_MIN_Y, DEEPLAB_MAX_Y, DEEPLAB_IMAGE_WIDTH
from yolo_bbox import YoloBBox
from nn_labels import DeeplabLabel, YoloLabel, DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID

def read_result_json(filename):
    if not os.path.isfile(filename):
        logging.error("File not found: %s", filename)
        return []
    with io.open(filename, encoding="utf-8") as _fp:
        jdata = json.load(_fp)
    return jdata


def _within_bboxes(yolo_bboxes, deeplab_class_id, deeplab_row, deeplab_col):
    """
    Return True if one of yolo_bboxes covers deeplab's result. False otherwise.
    """
    _x, _y = to_raw_image_pos(deeplab_col, deeplab_row)
    expected_id = DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID[deeplab_class_id]
    ret = False
    for bbox in yolo_bboxes:
        if bbox.class_id == expected_id:
            if bbox.is_within(_x, _y):
                ret = True
    return True


def _cmpr_yolo_with_deeplab(yolo_frame):
    """
    Return number of mismatches if yolo disagrees with deeplab or vice versa.
    Return 0 otherwise (that is, they have the same detection result).
    """
    num_mismatch = 0
    filename = yolo_frame["filename"]
    deeplab_mgr = DeeplabMgr(filename[:-4] + "_deeplab.png")
    bboxes = [YoloBBox(_) for _ in yolo_frame["objects"]]
    # deeplab finds an object, but yolo does not:
    for row in range(DEEPLAB_MIN_Y, DEEPLAB_MAX_Y):
        for col in range(DEEPLAB_IMAGE_WIDTH):
            deeplab_label = deeplab_mgr.get_label_by_xy(col, row)
            if deeplab_label == DeeplabLabel.BACKGROUND:
                continue
            if deeplab_label not in DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID:
                continue
            within = _within_bboxes(bboxes, deeplab_label, row, col)
            if not within:
                num_mismatch += 1
    # yolo finds an object, but deeplab does not:
    return num_mismatch


class YoloMgr(object):
    def __init__(self, json_file):
        self.frames = read_result_json(json_file)

    def find_weakness_images(self):
        for frame in self.frames:
            num_mismatch = _cmpr_yolo_with_deeplab(frame)
            logging.warning("Inspect %s", frame["filename"])
            if num_mismatch > 0:
                logging.warning("Find weakness: %s", frame["filename"])


if __name__ == "__main__":
    mgr = YoloMgr("/tmp/yolo_result.json")
    mgr.find_weakness_images()
