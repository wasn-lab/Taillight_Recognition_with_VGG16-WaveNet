#!/usr/bin/env python
import logging
from json_utils import read_json_file
from bbox import BBox
from nn_labels import EFFICIENTDET_CLASS_ID_TO_NAME, EfficientDetLabel


class EfficientDetMgr(object):
    def __init__(self, json_file):
        detection_result = read_json_file(json_file)
        self.bboxes = []
        for i in range(len(detection_result["class_ids"])):
            bbox = BBox()
            bbox.class_id = detection_result["class_ids"][i]
            if bbox.class_id not in EFFICIENTDET_CLASS_ID_TO_NAME:
                logging.warning("Ignore class_id %d", bbox.class_id)
                continue
            if bbox.class_id == EfficientDetLabel.TRUCK:
                bbox.class_id = EfficientDetLabel.CAR
            bbox.confidence = detection_result["scores"][i]
            roi = detection_result["rois"][i]
            bbox.left_x = int(roi[0] + 0.5)
            bbox.top_y = int(roi[1] + 0.5)
            bbox.right_x = int(roi[2] + 0.5)
            bbox.bottom_y = int(roi[3] + 0.5)
            bbox.name = EFFICIENTDET_CLASS_ID_TO_NAME[bbox.class_id]
            self.bboxes.append(bbox)

    def get_bboxes(self):
        return self.bboxes
