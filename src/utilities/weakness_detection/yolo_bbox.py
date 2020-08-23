#!/usr/bin/env python
from image_consts import RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT
from bbox import BBox
from nn_labels import YoloLabel


def gen_bbox_by_yolo_object(yolo_object, img_width=RAW_IMAGE_WIDTH, img_height=RAW_IMAGE_HEIGHT):
    """
    (0, 0) is the left-top corner of image.
    """
    relative_coordinates = yolo_object["relative_coordinates"]
    box = BBox()
    box.left_x, box.top_y, box.right_x, box.bottom_y = yolo_format_in_bbox(
        relative_coordinates["center_x"],
        relative_coordinates["center_y"],
        relative_coordinates["width"],
        relative_coordinates["height"],
        img_width,
        img_height)
    box.name = yolo_object["name"]
    box.class_id = yolo_object["class_id"]
    # treat truck as car because NN is easy to confuse them
    if box.class_id == YoloLabel.TRUCK:
        box.class_id = YoloLabel.CAR
    box.confidence = yolo_object["confidence"]
    return box


def yolo_format_in_bbox(center_x, center_y, width_x, height_y, img_width, img_height):
    _cx = center_x * img_width
    _cy = center_y * img_height
    bwidth = width_x * img_width
    bheight = height_y * img_height
    left_x = max(0, _cx - bwidth / 2)
    top_y = max(0, _cy - bheight / 2)
    right_x = min(img_width - 1, _cx + bwidth / 2)
    bottom_y = min(img_height - 1, _cy + bheight / 2)
    return int(left_x), int(top_y), int(right_x), int(bottom_y)


def bbox_in_yolo_format(left_x, top_y, right_x, bottom_y, img_width, img_height):
    left_x_frac = float(left_x) / img_width
    top_y_frac = float(top_y) / img_height
    right_x_frac = float(right_x) / img_width
    bottom_y_frac = float(bottom_y) / img_height
    return [(left_x_frac + right_x_frac) / 2,
            (top_y_frac + bottom_y_frac) / 2,
            right_x_frac - left_x_frac,
            bottom_y_frac - top_y_frac]
