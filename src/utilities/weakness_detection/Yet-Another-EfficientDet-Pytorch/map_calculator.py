# author: ICL-U
"""
Calculate mean average precision using EfficientDet-Pytorch
"""
import argparse
import io
import logging
import time
import sys
import subprocess
import os

from efficientdet_itri import EfficientDet, DRIVENET_CLASS_IDS

REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
REPO_DIR = REPO_DIR.decode("utf-8").strip()
UTILITIES_IOU_DIR = os.path.join(REPO_DIR, "src", "utilities", "iou")
sys.path.append(UTILITIES_IOU_DIR)

from iou_utils import calc_iou  # pylint: disable=import-error

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


class MAPCalculater():
    def __init__(self, coef, image_filenames, conf_thresh, iou_threshold):
        self.iou_threshold = iou_threshold
        self.edet = EfficientDet(coef, conf_thresh=conf_thresh,
                                 save_detection_result=False,
                                 save_yolo_fmt=False)
        with io.open(image_filenames, encoding="utf-8") as _fp:
            contents = _fp.read()
            self.image_list = [line.strip() for line in contents.splitlines()]

    def get_edet_bboxes(self):
        bboxes = []
        pred = self.edet.pred
        nobjs = len(pred["rois"])
        for j in range(nobjs):
            class_id = pred['class_ids'][j]
            if class_id not in DRIVENET_CLASS_IDS:
                continue
            box = [class_id]
            for item in pred['rois'][j]:
                box.append(int(item + 0.5))
            bboxes.append(box)
        return bboxes

    def get_ground_truth(self, image_filename):
        txt_filename = image_filename[:-4] + ".txt"
        img_width, img_height = self.edet.img.shape[1], self.edet.img.shape[0]
        bboxes = []
        with io.open(txt_filename, encoding="utf-8") as _fp:
            contents = _fp.read()
        for line in contents.splitlines():
            fields = line.strip().split()
            class_id = int(fields[0])
            cx = float(fields[1])
            cy = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])

            left_x, top_y, right_x, bottom_y = yolo_format_in_bbox(
                cx, cy, width, height, img_width, img_height)
            bboxes.append([class_id, left_x, top_y, right_x, bottom_y])
        return bboxes

    def calc_tp_fp_fn(self, gt_boxes, edet_boxes):
        gt_match = [0 for i in range(len(gt_boxes))]
        edet_match = [0 for i in range(len(edet_boxes))]
        for i in range(len(gt_boxes)):
            logging.debug("Inspect gt_boxes[%d]: %s", i, gt_boxes[i])
            if gt_match[i] > 0:
                continue
            max_j = -1
            for j in range(len(edet_boxes)):
                if edet_match[j] > 0:
                    continue
                iou = calc_iou(gt_boxes[i], edet_boxes[j])
                logging.debug("iou with %s: %f", edet_boxes[j], iou)
                if iou < self.iou_threshold:
                    continue
                max_j = max(j, max_j)
            if max_j >= 0:
                logging.debug("%s match %s", gt_boxes[i], edet_boxes[j])
                gt_match[i] = 1
                edet_match[max_j] = 1
        true_positive = sum(edet_match)
        false_positive = len(edet_boxes) - true_positive
        false_negative = len(gt_boxes) - sum(gt_match)
        assert true_positive >= 0
        assert false_positive >= 0
        assert false_negative >= 0
        return true_positive, false_positive, false_negative

    def run(self):

        start_time = time.time()
        total_tp, total_fp, total_fn = 0, 0, 0
        for image_filename in self.image_list:
            self.edet.inference_by_file(image_filename)
            gt_boxes = self.get_ground_truth(image_filename)
            edet_boxes = self.get_edet_bboxes()
            _tp, _fp, _fn = self.calc_tp_fp_fn(gt_boxes, edet_boxes)
            logging.warning("TP: %d, FP: %d, FN: %d", _tp, _fp, _fn)
            total_tp += _tp
            total_fp += _fp
            total_fn += _fn
        print("-" * 40)
        print("Summary:")
        print("EfficientNet d{}: confidence threshold: {}".format(
            self.edet.compound_coef, self.edet.conf_thresh))
        print("IOU threshold: {}".format(self.iou_threshold))
        print("TP: {}, FP: {}, FN: {}".format(total_tp, total_fp, total_fn))
        acc = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        print("Precision: {:f}, Recall: {:f}".format(acc, recall))
        print("Total execution time: {}s".format(time.time() - start_time))
        print("-" * 40)

def main():

    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef", type=int, default=4, help="efficientnet coefficient")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--image-filenames", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    mcalc = MAPCalculater(args.coef, args.image_filenames, args.conf_thresh, args.iou_threshold)
    mcalc.run()


if __name__ == "__main__":
    main()
