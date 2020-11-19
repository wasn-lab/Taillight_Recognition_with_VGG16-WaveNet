import logging
def calc_iou(box1, box2):
    """box: [left_x, top_y, right_x, bottom_y]"""
    left_x = max(box1[0], box2[0])
    right_x = min(box1[2], box2[2])
    top_y = max(box1[1], box2[1])
    bottom_y = min(box1[3], box2[3])
    if left_x >= right_x or top_y >= bottom_y:
        return 0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intersection = (right_x - left_x) * (bottom_y - top_y)
    return float(intersection) / (area1 + area2 - intersection)


def calc_iou5(box1, box2):
    """box: [class_id, left_x, top_y, right_x, bottom_y]"""
    if box1[0] != box2[0]:
        return 0
    return calc_iou(box1[1:], box2[1:])
