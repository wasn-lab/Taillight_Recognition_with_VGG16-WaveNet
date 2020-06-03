#!/usr/bin/env python
import os
import io
import json
import logging
import pprint
from deeplab_mgr import DeeplabMgr, deeplab_pos_to_raw_pos, raw_image_pos_to_deeplab_pos
from image_consts import DEEPLAB_MIN_Y, DEEPLAB_MAX_Y, DEEPLAB_IMAGE_WIDTH
from yolo_bbox import YoloBBox
from nn_labels import DeeplabLabel, DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID, YOLO_CLASS_ID_TO_DEEPLAB_CLASS_ID

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
    _x, _y = deeplab_pos_to_raw_pos(deeplab_col, deeplab_row)
    expected_id = DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID[deeplab_class_id]
    ret = False
    for bbox in yolo_bboxes:
        if bbox.class_id == expected_id:
            if bbox.is_within(_x, _y):
                ret = True
    return ret


def _yolo_contains_enough_deeplab_labels(bbox, deeplab_mgr):
    """Return True if |bbox| contains 20% or more deeplab detections."""
    if bbox.class_id not in YOLO_CLASS_ID_TO_DEEPLAB_CLASS_ID:
        return True
    num_pixels_with_same_id = 0
    expected_id = YOLO_CLASS_ID_TO_DEEPLAB_CLASS_ID[bbox.class_id]
    for row in range(bbox.top_y, bbox.bottom_y + 1):
        for col in range(bbox.left_x, bbox.right_x + 1):
            deeplab_x, deeplab_y = raw_image_pos_to_deeplab_pos(col, row)
            if deeplab_mgr.get_label_by_xy(deeplab_x, deeplab_y) == expected_id:
                num_pixels_with_same_id += 1
    total_bbox_pixels = (bbox.bottom_y - bbox.top_y + 1) * (bbox.right_x - bbox.left_x + 1)
    return bool(num_pixels_with_same_id > total_bbox_pixels * 0.2)


def _cmpr_yolo_with_deeplab(yolo_frame):
    """
    Return number of mismatches if yolo disagrees with deeplab or vice versa.
    Return 0 otherwise (that is, they have the same detection result).
    """
    num_mismatch = 0
    filename = yolo_frame["filename"]
    deeplab_mgr = DeeplabMgr(filename[:-4] + "_deeplab_labels.png")
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
    if num_mismatch > 0:
        logging.warning("Deeplab finds object but yolo does not: %s", filename)
    # yolo finds an object, but deeplab does not:
    for bbox in bboxes:
        if not _yolo_contains_enough_deeplab_labels(bbox, deeplab_mgr):
            logging.warning("Yolo finds object but deeplab does not: %s", filename)
            num_mismatch += 1
    return num_mismatch


class YoloMgr(object):
    def __init__(self, json_file):
        self.frames = read_result_json(json_file)

    def find_weakness_images(self):
        num_weak_images_found = 0
        for frame in self.frames:
            frame["deeplab_disagree"] = _cmpr_yolo_with_deeplab(frame)
        self.frames.sort(key=lambda x: x["deeplab_disagree"])
        for frame in self.frames[-10:]:
            pprint.pprint(frame)

if __name__ == "__main__":
    mgr = YoloMgr("/tmp/yolo_result.json")
    mgr.find_weakness_images()
