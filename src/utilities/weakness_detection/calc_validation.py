# author: ICL-U
"""
Validate the weakness detection.
"""
import argparse
import io
import logging
import sys
import subprocess
import os
import shutil

from deeplab_mgr import DeeplabMgr, raw_image_pos_to_deeplab_pos
from json_utils import read_json_file
from image_utils import get_image_size
from nn_labels import (DRIVENET_CLASS_IDS, DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID,
                       YoloLabel)
from yolo_bbox import yolo_format_in_bbox

REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
REPO_DIR = REPO_DIR.decode("utf-8").strip()
UTILITIES_IOU_DIR = os.path.join(REPO_DIR, "src", "utilities", "iou")
sys.path.append(UTILITIES_IOU_DIR)

from iou_utils import calc_iou5  # pylint: disable=import-error


class Validator():
    def __init__(self, yolo_result_json, coef, weak_image_list, iou_threshold, with_deeplab, save_files, with_roi):
        self.iou_threshold = iou_threshold
        self.coef = coef
        self.save_files = save_files
        self.with_deeplab = with_deeplab
        self.with_roi = with_roi
        self.yolo_result = self.get_yolo_result(yolo_result_json)
        with io.open(weak_image_list, encoding="utf-8") as _fp:
            contents = _fp.read()
            self.weak_images = [_.strip() for _ in contents.splitlines()]

    def get_yolo_result(self, json_file):
        yolo_result = {}
        for doc in read_json_file(json_file):
            filename = doc["filename"]
            img_width, img_height = get_image_size(filename)
            yolo_result[filename] = []
            for obj in doc["objects"]:
                class_id = obj["class_id"]
                if class_id not in DRIVENET_CLASS_IDS:
                    continue
                bbox = [class_id]
                bbox += yolo_format_in_bbox(
                    obj["relative_coordinates"]["center_x"],
                    obj["relative_coordinates"]["center_y"],
                    obj["relative_coordinates"]["width"],
                    obj["relative_coordinates"]["height"],
                    img_width, img_height)
                yolo_result[filename].append(bbox)
        return yolo_result

    def get_deeplab_results(self, filename):
        png = filename[:-4] + "_deeplab_labels.png"
        return DeeplabMgr(png)

    def get_yolo_bboxes(self, filename):
        return self.yolo_result[filename]

    def get_edet_bboxes(self, filename):
        bboxes = []
        pred = read_json_file(filename[:-4] + "_efficientdet_d{}.json".format(self.coef))
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

    def get_gt_bboxes(self, image_filename):
        txt_filename = image_filename[:-4] + ".txt"
        img_width, img_height = get_image_size(image_filename)
        bboxes = []
        with io.open(txt_filename, encoding="utf-8") as _fp:
            contents = _fp.read()
        for line in contents.splitlines():
            fields = line.strip().split()
            class_id = int(fields[0])
            _cx = float(fields[1])
            _cy = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])

            left_x, top_y, right_x, bottom_y = yolo_format_in_bbox(
                _cx, _cy, width, height, img_width, img_height)
            bboxes.append([class_id, left_x, top_y, right_x, bottom_y])
        return bboxes

    def within_roi(self, image_filename, x, y):
        #    ---------
        #    |   P1  |
        #    |  ---  |
        #    | /   \ |
        #    |/     \|
        # P0 |       |P2
        #    ---------
        img_width, img_height = get_image_size(image_filename)
        bottom_y_ratio = 1000.0 / 1208
        top_y_ratio = 800.0 / 1208
        if img_width == 1280:
            p0_x = 560
            p0_y = img_height - 1
            p1_x = img_width / 2
            p1_y = 300
            p2_x = 1550
            p2_y = img_height - 1
        else:
            p0_x = 0
            p0_y = int(img_height * bottom_y_ratio)
            p1_x = img_width / 2
            p1_y = int(img_height * top_y_ratio)
            p2_x = img_width - 1
            p2_y = p0_y

        k1 = (p1_y - p0_y)*(x - p0_x) - (p1_x - p0_x) * (y - p0_y)
        k2 = (p1_y - p2_y)*(x - p2_x) - (p1_x - p2_x) * (y - p2_y)
        logging.debug("p0 = (%d, %d) p1 = (%d, %d) p2= (%d, %d), x=%d, y=%d, k1=%d, k2=%d",
                      p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, x, y, k1, k2)
        if k1 < 0 and k2 > 0:
            return True
        return False

    def bbox_within_roi(self, image_filename, gt_box):
        if gt_box[0] == YoloLabel.PERSON:
            return True
        return False
        left_x, top_y, right_x, bottom_y = gt_box[1:]
        return (self.within_roi(image_filename, left_x, top_y) and
                self.within_roi(image_filename, right_x, top_y) and
                self.within_roi(image_filename, left_x, bottom_y) and
                self.within_roi(image_filename, right_x, bottom_y))

    def run(self):
        all_tp = 0
        all_fp = 0
        all_fn = 0
        records = []
        for filename in self.weak_images:
            if self.with_deeplab:
                true_positive, false_positive, false_negative = self.calc_tp_fp_fn(filename)
            else:
                true_positive, false_positive, false_negative = self.calc_tp_fp_fn_only_edet(filename)
            if self.save_files and true_positive + false_positive + false_negative > 0:
                dest = "/tmp"
                logging.warn("cp %s", filename)
                shutil.copy(filename, dest)
                shutil.copy(filename[:-4] + ".txt", dest)
            all_tp += true_positive
            all_fp += false_positive
            all_fn += false_negative
            records.append("{},{},{},{}".format(true_positive, false_positive, false_negative, filename))
        logging.info("TP: %d, FP: %d, FN: %d", all_tp, all_fp, all_fn)
        precision = float(all_tp) / (all_tp + all_fp)
        recall = float(all_tp) / (all_tp + all_fn)
        logging.info("precision: %f, recall: %f", precision, recall)
        with io.open("records.log", "w") as _fp:
            _fp.write("\n".join(records))
            _fp.write("\n")
        logging.warning("Write records.log")

    def calc_tp_fp_fn_only_edet(self, filename):
        yolo_bboxes = self.get_yolo_bboxes(filename)
        edet_bboxes = self.get_edet_bboxes(filename)
        gt_bboxes = self.get_gt_bboxes(filename)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        img_width, img_height = get_image_size(filename)
        for gt_bbox in gt_bboxes:
            if self.with_roi and not self.bbox_within_roi(filename, gt_bbox):
                logging.info("Skip gtbox %s", gt_bbox[1:])
                continue
            if self.with_roi:
                logging.info("gtbox (%d) in roi: %s", gt_bbox[0], gt_bbox[1:])
            yolo_match = False
            edet_match = False
            for yolo_bbox in yolo_bboxes:
                if calc_iou5(yolo_bbox, gt_bbox) >= self.iou_threshold:
                    logging.debug("Yolo match: %s with %s", yolo_bbox, gt_bbox)
                    yolo_match = True
                    break
            for edet_bbox in edet_bboxes:
                if calc_iou5(edet_bbox, gt_bbox) >= self.iou_threshold:
                    logging.debug("Edet match: %s with %s", edet_bbox, gt_bbox)
                    edet_match = True
                    break
            if yolo_match and not edet_match:
                false_positive += 1
            if not yolo_match:
                if edet_match:
                    true_positive += 1
                else:
                    false_negative += 1
        logging.info("%s (%dx%d): true_positive: %d, false_positive:%d, false_negative: %d, groundtruth: %d",
                     filename, img_width, img_height, true_positive, false_positive, false_negative, len(gt_bboxes))
        return true_positive, false_positive, false_negative

    def calc_tp_fp_fn(self, filename):
        yolo_bboxes = self.get_yolo_bboxes(filename)
        edet_bboxes = self.get_edet_bboxes(filename)
        gt_bboxes = self.get_gt_bboxes(filename)
        deeplab_mgr = self.get_deeplab_results(filename)
        true_positive = 0
        false_negative = 0
        false_positive = 0
        img_width, img_height = get_image_size(filename)
        for gt_bbox in gt_bboxes:
            if self.with_roi and not self.bbox_within_roi(filename, gt_bbox):
                logging.info("Skip gtbox %s", gt_bbox[1:])
                continue
            if self.with_roi:
                logging.info("gtbox (%d) in roi: %s", gt_bbox[0], gt_bbox[1:])
            yolo_match = False
            edet_match = False
            deeplab_match = False
            for yolo_bbox in yolo_bboxes:
                if calc_iou5(yolo_bbox, gt_bbox) >= self.iou_threshold:
                    logging.debug("Yolo match: %s with %s", yolo_bbox, gt_bbox)
                    yolo_match = True
                    break
            for edet_bbox in edet_bboxes:
                if calc_iou5(edet_bbox, gt_bbox) >= self.iou_threshold:
                    logging.debug("Edet match: %s with %s", edet_bbox, gt_bbox)
                    edet_match = True
                    break
            for y in range(gt_bbox[2], gt_bbox[4]):
                if deeplab_match:
                    break
                for x in range(gt_bbox[1], gt_bbox[3]):
                    deeplab_x, deeplab_y = raw_image_pos_to_deeplab_pos(x, y, img_width, img_height)
                    class_id = deeplab_mgr.get_label_by_xy(deeplab_x, deeplab_y)
                    if class_id not in DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID:
                        continue
                    if DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID[class_id] == gt_bbox[0]:
                        logging.debug("Deeplab match at (%d, %d) for gt_box %s", x, y, gt_bbox)
                        deeplab_match = True
                        break
            if yolo_match and (not edet_match) and (not deeplab_match):
                false_positive += 1

            if not yolo_match:
                if edet_match and deeplab_match:
                    true_positive += 1
                else:
                    false_negative += 1

        logging.info("%s (%dx%d): true_positive: %d, false_positive:%d, false_negative: %d, groundtruth: %d",
                     filename, img_width, img_height, true_positive, false_positive, false_negative, len(gt_bboxes))
        return true_positive, false_positive, false_negative

def main():

    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-result-json", required=True)
    parser.add_argument("--coef", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--weak-image-list", required=True)
    parser.add_argument("--with-deeplab", action="store_true")
    parser.add_argument("--save-files", action="store_true")
    parser.add_argument("--with-roi", action="store_true")
    args = parser.parse_args()
    obj = Validator(args.yolo_result_json, args.coef, args.weak_image_list,
                    args.iou_threshold, args.with_deeplab, args.save_files, args.with_roi)
    obj.run()


if __name__ == "__main__":
    main()
