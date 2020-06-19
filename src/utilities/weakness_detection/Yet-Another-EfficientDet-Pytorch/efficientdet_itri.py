# Author: Zylo117, modified by ICL-U
"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import logging
import torch
import argparse
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

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
THRESHOLD = 0.2
IOU_THRESHOLD = 0.2


def get_input_size():
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[COMPOUND_COEF]
    return input_size

def get_model(cache=list()):
    if cache:
        return cache[0]

    logging.warning("Init model")

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]


    use_cuda = True
    use_float16 = False
    cudnn.fastest = True


    model = EfficientDetBackbone(compound_coef=COMPOUND_COEF, num_classes=len(OBJ_LIST),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load('weights/efficientdet-d{}.pth'.format(COMPOUND_COEF)))
    model.requires_grad_(False)
    model.eval()

    model = model.cuda()
    cache.append(model)
    return model

def preprocess_image(img_path):
    # tf bilinear interpolation is different from any other's, just make do
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=get_input_size())
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    x = x.to(torch.float32).permute(0, 3, 1, 2)
    return x, framed_metas


def inference(img_path):
    pimg, framed_metas = preprocess_image(img_path)
    model = get_model()

    with torch.no_grad():
        features, regression, classification, anchors = model(pimg)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(pimg,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          THRESHOLD, IOU_THRESHOLD)

        out = invert_affine(framed_metas, out)
