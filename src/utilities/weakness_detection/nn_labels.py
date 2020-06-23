#!/usr/bin/env python
"""Definition of labels of neural networks"""



class DeeplabLabel(object):
    BACKGROUND = 0
    BUS = 6
    CAR = 7
    MOTORBIKE = 14
    PERSON = 15


class YoloLabel(object):
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORBIKE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9

class EfficientDetLabel(object):
    """Same as Yolo. Other labels in efficientdet are ignored."""
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORBIKE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9

DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID = {
    DeeplabLabel.BUS: YoloLabel.BUS,
    DeeplabLabel.CAR: YoloLabel.CAR,
    DeeplabLabel.MOTORBIKE: YoloLabel.MOTORBIKE,
    DeeplabLabel.PERSON: YoloLabel.PERSON}

YOLO_CLASS_ID_TO_DEEPLAB_CLASS_ID = {
    YoloLabel.BUS: DeeplabLabel.BUS,
    YoloLabel.CAR: DeeplabLabel.CAR,
    YoloLabel.MOTORBIKE: DeeplabLabel.MOTORBIKE,
    YoloLabel.PERSON: DeeplabLabel.PERSON,
    }
