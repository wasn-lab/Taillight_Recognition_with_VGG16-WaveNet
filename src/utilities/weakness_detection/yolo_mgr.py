#!/usr/bin/env python
import os
import json
import logging
import datetime
import sys
import subprocess
import multiprocessing
from deeplab_mgr import DeeplabMgr, deeplab_pos_to_raw_pos, raw_image_pos_to_deeplab_pos
from image_consts import DEEPLAB_MIN_Y, DEEPLAB_MAX_Y, DEEPLAB_IMAGE_WIDTH
from yolo_bbox import gen_bbox_by_yolo_object
from nn_labels import DeeplabLabel, DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID, YOLO_CLASS_ID_TO_DEEPLAB_CLASS_ID
from json_utils import read_json_file
from efficientdet_mgr import EfficientDetMgr

REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
REPO_DIR = REPO_DIR.decode("utf-8").strip()
UTILITIES_IOU_DIR = os.path.join(REPO_DIR, "src", "utilities", "iou")
sys.path.append(UTILITIES_IOU_DIR)

from iou_utils import calc_iou  # pylint: disable=import-error


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


def _deeplab_covered_by_enough_bboxes(bboxes, deeplab_mgr, filename):
    # deeplab finds an object, but yolo does not:
    total_object_labels = 0
    num_labels_covered = 0
    for row in range(DEEPLAB_MIN_Y, DEEPLAB_MAX_Y):
        for col in range(DEEPLAB_IMAGE_WIDTH):
            deeplab_label = deeplab_mgr.get_label_by_xy(col, row)
            if deeplab_label == DeeplabLabel.BACKGROUND:
                continue
            if deeplab_label not in DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID:
                continue
            total_object_labels += 1
            if _within_bboxes(bboxes, deeplab_label, row, col):
                num_labels_covered += 1
    if num_labels_covered >= total_object_labels * 0.9:
        return 0
    logging.warning("Deeplab is not covered by yolo enough: %s", filename)
    return 1


def _cmpr_yolo_with_deeplab(yolo_frame):
    """
    Return number of mismatches if yolo disagrees with deeplab or vice versa.
    Return 0 otherwise (that is, they have the same detection result).
    """
    filename = yolo_frame["filename"]
    deeplab_mgr = DeeplabMgr(filename[:-4] + "_deeplab_labels.png")
    bboxes = [gen_bbox_by_yolo_object(_) for _ in yolo_frame["objects"]]
    num_mismatch = _deeplab_covered_by_enough_bboxes(bboxes, deeplab_mgr, filename)
    # yolo finds an object, but deeplab does not:
    for bbox in bboxes:
        if not _yolo_contains_enough_deeplab_labels(bbox, deeplab_mgr):
            logging.warning("Yolo finds object but deeplab does not: %s", filename)
            num_mismatch += 1
    return num_mismatch


def _cmpr_yolo_with_efficientdet(yolo_frame):
    filename = yolo_frame["filename"]
    edet_mgr = EfficientDetMgr(filename[:-4] + "_efficientdet_d4.json")
    yolo_bboxes = [gen_bbox_by_yolo_object(_) for _ in yolo_frame["objects"]]
    edet_bboxes = edet_mgr.get_bboxes()
    num_mismatch = abs(len(yolo_bboxes) - len(edet_bboxes))
    for ybox in yolo_bboxes:
        match = False
        for ebox in edet_bboxes:
            if ebox.class_id != ybox.class_id:
                continue
            iou = calc_iou(ybox, ebox)
            if iou >= 0.25:
                match = True
        if not match:
            num_mismatch += 1

    for ebox in edet_bboxes:
        match = False
        for ybox in yolo_bboxes:
            if ebox.class_id != ybox.class_id:
                continue
            iou = calc_iou(ybox, ebox)
            if iou >= 0.25:
                match = True
        if not match:
            num_mismatch += 1
    return num_mismatch


class YoloMgr(object):
    def __init__(self, json_file):
        self.frames = read_json_file(json_file)

    def get_weakest_images(self, amount=0):
        """Move weakest images to |weakness_dir|"""
        self.frames.sort(key=lambda x: x["disagree_level"])
        if amount <= 0:
            amount = int(len(self.frames) * 0.05) + 1
        return [_["filename"] for _ in self.frames[-amount:] if _["disagree_level"] > 0]

    def save_weakness_logs(self, weakness_dir):
        now = datetime.datetime.now()
        filename = "{}{:02d}{:02d}{:02d}{:02d}.json".format(
            now.year, now.month, now.day, now.hour, now.minute)
        output_file = os.path.join(weakness_dir, filename)
        contents = json.dumps(self.frames, sort_keys=True)
        with open(output_file, "w") as _fp:
            _fp.write(contents)
        logging.warning("Write %s", output_file)

    def find_weakness_images(self):
        nproc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(nproc)

        res = pool.map(_cmpr_yolo_with_deeplab, self.frames)
        for idx, frame in enumerate(self.frames):
            frame["deeplab_disagree"] = res[idx]

        res = pool.map(_cmpr_yolo_with_efficientdet, self.frames)
        for idx, frame in enumerate(self.frames):
            frame["edet_disagree"] = res[idx]

        pool.close()
        pool.join()

        for frame in self.frames:
            frame["disagree_level"] = frame["edet_disagree"] * frame["deeplab_disagree"]
            logging.warning("%s: deeplab: %d, edet: %d, level: %d",
                frame["filename"], frame["deeplab_disagree"],
                frame["edet_disagree"], frame["disagree_level"])
