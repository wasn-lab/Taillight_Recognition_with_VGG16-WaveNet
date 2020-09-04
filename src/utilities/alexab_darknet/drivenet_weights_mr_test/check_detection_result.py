#!/usr/bin/env python
import argparse
import io
import json
import logging
import os
import sys
import pprint

from yolo_bbox import gen_bbox_by_yolo_object
from iou_utils import calc_iou5

def _read_json_file(jfile):
    with io.open(jfile, encoding="utf-8") as _fp:
        contents = _fp.read()
    return json.loads(contents)


def _get_expectation(filename):
    jfile = filename[:-4] + ".json"
    return _read_json_file(jfile)


def _calc_violations(actual, expectation):
    """
    Args:
    expectation: [
        {u'class_id': 0, u'iou': 0.9, u'negative': False,
         u'coordinates': [424, 227, 453, 344]}, ...]
    actual: [[2, 118, 255, 173, 288], [2, 445, 242, 531, 304],...]
    """
    nviolations = 0
    for expected_item in expectation:
        exp_box = [expected_item["class_id"]] + expected_item["coordinates"]
        match = False
        for abox in actual:
            iou = calc_iou5(exp_box, abox)
            if iou >= expected_item["iou"]:
                match = True
        if (not match) and (not expected_item["negative"]):
            logging.warn("BBox %s does not match", str(exp_box))

        if match and expected_item["negative"]:
            match = False
            logging.warn("BBox %s should not match", str(exp_box))
        if not match:
            nviolations += 1
    return nviolations


def _check_detection_result(result_json):
    jdata = _read_json_file(result_json)
    docs = []
    base_url = "http://ci.itriadv.co/"
    for doc in jdata:
        expectation = _get_expectation(doc["filename"])
        actual = [gen_bbox_by_yolo_object(_) for _ in doc["objects"]]
        nviolations = _calc_violations(actual, expectation)
        org_image_components = doc["filename"].split("/")[2:]
        org_image_url = base_url + "/".join(org_image_components)

        rdoc = {"filename": doc["filename"],
                "num_violations": nviolations,
                "expect": org_image_url[:-4] + "_expect.jpg",
                "actual": org_image_url[:-4] + "_yolo.jpg",
                "result": "PASS"}
        if nviolations > 0:
            logging.warn("%s: Unexpected detection result", doc["filename"])
            rdoc["result"] = "FAIL"
        docs.append(rdoc)
    return docs

def _write_check_result(docs, output_dir):
    output_file = os.path.join(output_dir, "check_weights_result.json")
    with io.open(output_file, "w", encoding="utf-8") as _fp:
        contents = json.dumps(docs, sort_keys=True)
        _fp.write(contents)
    logging.warning("Write %s", output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-result-json", required=True)
    parser.add_argument("--output-dir", default="/tmp", required=True)
    args = parser.parse_args()
    docs = _check_detection_result(args.yolo_result_json)
    _write_check_result(docs, args.output_dir)

if __name__ == "__main__":
    main()
