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

from deeplab_mgr import DeeplabMgr, raw_image_pos_to_deeplab_pos
from json_utils import read_json_file
from image_utils import get_image_size
from nn_labels import DRIVENET_CLASS_IDS, DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID
from yolo_bbox import yolo_format_in_bbox

REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
REPO_DIR = REPO_DIR.decode("utf-8").strip()
UTILITIES_IOU_DIR = os.path.join(REPO_DIR, "src", "utilities", "iou")
sys.path.append(UTILITIES_IOU_DIR)

from iou_utils import calc_iou5  # pylint: disable=import-error


class Validator():
    def __init__(self, yolo_result_json, weak_image_list, iou_threshold):
        self.iou_threshold = iou_threshold
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
        pred = read_json_file(filename[:-4] + "_efficientdet_d4.json")
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
            cx = float(fields[1])
            cy = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])

            left_x, top_y, right_x, bottom_y = yolo_format_in_bbox(
                cx, cy, width, height, img_width, img_height)
            bboxes.append([class_id, left_x, top_y, right_x, bottom_y])
        return bboxes

    def run(self):
        all_tp = 0
        all_fp = 0
        all_fn = 0
        for filename in self.weak_images:
            tp, fp, fn = self.calc_tp_fp_fn(filename)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        logging.info("TP: %d, FP: %d, FN: %d", all_tp, all_fp, all_fn)


    def calc_tp_fp_fn(self, filename):
        yolo_bboxes = self.get_yolo_bboxes(filename)
        edet_bboxes = self.get_edet_bboxes(filename)
        gt_bboxes = self.get_gt_bboxes(filename)
        deeplab_mgr = self.get_deeplab_results(filename)
        tp = 0
        fn = 0
        fp = 0
        img_width, img_height = get_image_size(filename)
        for gt_bbox in gt_bboxes:
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
#            for y in range(gt_bbox[2], gt_bbox[4]):
#                if deeplab_match:
#                    break
#                for x in range(gt_bbox[1], gt_bbox[3]):
#                    deeplab_x, deeplab_y = raw_image_pos_to_deeplab_pos(x, y, img_width, img_height)
#                    class_id = deeplab_mgr.get_label_by_xy(deeplab_x, deeplab_y)
#                    if class_id not in DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID:
#                        continue
#                    if DEEPLAB_CLASS_ID_TO_YOLO_CLASS_ID[class_id] == gt_bbox[0]:
#                        logging.debug("Deeplab match at (%d, %d) for gt_box %s", x, y, gt_bbox)
#                        deeplab_match = True
#                        break
#            if yolo_match and (not edet_match) and (not deeplab_match):
#                fp += 1
#
#            if not yolo_match:
#                if edet_match and deeplab_match:
#                    tp += 1
#                else:
#                    fn += 1
            if yolo_match and not edet_match:
                fp += 1
            if not yolo_match:
                if edet_match:
                    tp += 1
                else:
                    fn += 1
        logging.info("%s (%dx%d): tp: %d, fp:%d, fn: %d, groundtruth: %d",
                     filename, img_width, img_height, tp, fp, fn, len(gt_bboxes))
        return tp, fp, fn

def main():

    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-result-json", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--weak-image-list", required=True)
    args = parser.parse_args()
    obj = Validator(args.yolo_result_json, args.weak_image_list, args.iou_threshold)
    obj.run()


if __name__ == "__main__":
    main()
