# Author: Zylo117, modified by ICL-U
"""
Simple Inference Script of EfficientDet-Pytorch
"""
import argparse
import json
import io
import logging
import os
import time

import torch
#from torch.backends import cudnn
#from matplotlib import colors
import numpy as np
import cv2

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (preprocess, invert_affine, postprocess,
                         STANDARD_COLORS, standard_to_bgr, get_index_label,
                         plot_one_box)


COMPOUND_COEF = 4
OBJ_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', '', 'dining table', '', '',
            'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
COLOR_LIST = standard_to_bgr(STANDARD_COLORS)
THRESHOLD = 0.3
IOU_THRESHOLD = 0.2
DRIVENET_CLASS_IDS = [0,1,2,3,5,7] # only output these class

def bbox_in_yolo_format(left_x, top_y, right_x, bottom_y, img_width, img_height):
    left_x_frac = float(left_x) / img_width
    top_y_frac = float(top_y) / img_height
    right_x_frac = float(right_x) / img_width
    bottom_y_frac = float(bottom_y) / img_height
    return [(left_x_frac + right_x_frac) / 2,
            (top_y_frac + bottom_y_frac) / 2,
            right_x_frac - left_x_frac,
            bottom_y_frac - top_y_frac]

class EfficientDet():
    """Object detection using efficientDet"""
    def __init__(self, coef):
        self.model = None
        self.compound_coef = coef

    def get_input_size(self):
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        input_size = input_sizes[self.compound_coef]
        return input_size

    def get_model(self):
        if self.model:
            return self.model

        logging.warning("Init model")
        # replace this part with your project's anchor config
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(OBJ_LIST),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load('weights/efficientdet-d{}.pth'.format(self.compound_coef)))
        model.requires_grad_(False)
        model.eval()

        model = model.cuda()
        self.model = model
        return model

    def preprocess_image(self, img_path):
        # tf bilinear interpolation is different from any other's, just make do
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=self.get_input_size())
        _x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        _x = _x.to(torch.float32).permute(0, 3, 1, 2)
        return _x, ori_imgs, framed_metas

    def save_pred(self, pred, img, img_path):
        img_height, img_width = img.shape[0], img.shape[1]
        nobjs = len(pred["rois"])
        boxes = []
        for j in range(nobjs):
            class_id = pred['class_ids'][j]
            if class_id not in DRIVENET_CLASS_IDS:
                continue
            left_x, top_y, right_x, bottom_y = pred['rois'][j]
            cx, cy, width, height = bbox_in_yolo_format(
                left_x, top_y, right_x, bottom_y, img_width, img_height)
            boxes.append([class_id, cx, cy, width, height])
        output_file = img_path[:-4] + ".txt"
        with io.open(output_file, "w", encoding="utf-8") as _fp:
            for box in boxes:
                line = []
                for item in box:
                    line.append("{}".format(item))

                _fp.write(" ".join(line))
                _fp.write("\n");
        logging.warning("Write %s", output_file)

    def gen_yolo_bbox(self, img_path):
        start_time = time.time()
        logging.warning("Load %s", img_path)
        pimg, ori_imgs, framed_metas = self.preprocess_image(img_path)
        model = self.get_model()

        with torch.no_grad():
            __features, regression, classification, anchors = model(pimg)
            regress_boxes = BBoxTransform()
            clip_boxes = ClipBoxes()
            out = postprocess(pimg,
                              anchors, regression, classification,
                              regress_boxes, clip_boxes,
                              THRESHOLD, IOU_THRESHOLD)
            out = invert_affine(framed_metas, out)
            self.save_pred(out[0], ori_imgs[0], img_path)
        logging.info("Inference time: %f", time.time() - start_time)

def main():
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef", type=int, default=4)
    parser.add_argument("--image-filenames", required=True)

    args = parser.parse_args()
    edet = EfficientDet(args.coef)
    with io.open(args.image_filenames, encoding="utf-8") as _fp:
        contents = _fp.read()
    for line in contents.splitlines():
        line = line.strip()
        edet.gen_yolo_bbox(line)


if __name__ == "__main__":
    main()
