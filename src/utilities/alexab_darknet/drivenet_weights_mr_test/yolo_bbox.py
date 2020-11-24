#!/usr/bin/env python


def gen_bbox_by_yolo_object(yolo_object):
    """
    (0, 0) is the left-top corner of image.
    """
    # TODO: detect image width/height
    img_width = 608
    img_height = 384
    relative_coordinates = yolo_object["relative_coordinates"]
    abs_coordinates = yolo_format_in_bbox(
        relative_coordinates["center_x"],
        relative_coordinates["center_y"],
        relative_coordinates["width"],
        relative_coordinates["height"],
        img_width,
        img_height)
    return [yolo_object["class_id"]] + list(abs_coordinates)


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
