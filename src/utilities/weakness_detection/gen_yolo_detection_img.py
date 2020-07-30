#!/usr/bin/env python3
"""
Generate the detection results of images. The output include a json file and
images drawn with bounding boxes.
"""
import argparse
import os
import logging
import io
import shutil
import pexpect

WEAKNESS_DETECTION_DIR = os.path.join(os.path.abspath(__file__), "..")
REPO_DIR = os.path.normpath(os.path.join(WEAKNESS_DETECTION_DIR, "..", "..", ".."))
DARKNET_DIR = os.path.join(REPO_DIR, "src/utilities/alexab_darknet")
DRIVENET_DIR = os.path.join(REPO_DIR, "src/sensing/itri_drivenet/drivenet")


def process_filenames_arg(txt_file):
    """Return a list of image file names shown in |txt_file|."""
    with io.open(txt_file, encoding="utf-8") as _fp:
        contents = _fp.read()
    return [_.strip() for _ in contents.splitlines()]


def detect_by_yolo(darknet_exe, yolo_data_file, yolo_cfg_file, yolo_weights,
                   json_output, image_filenames):
    os.chdir(DARKNET_DIR)
    cmd = [darknet_exe, "detector", "test", yolo_data_file, yolo_cfg_file,
           yolo_weights, "-thresh", "0.5", "-dont_show", "-ext_output",
           "-out", json_output]
    print(" ".join(cmd))
    child = pexpect.spawnu(" ".join(cmd))
    child.expect("Enter Image Path:")
    print(child.before)
    print(child.after)

    default_img_with_detection = os.path.join(DARKNET_DIR, "predictions.jpg")

    for filename in image_filenames:
        out = filename[:-4] + "_yolo.jpg"
        child.sendline(filename)
        child.expect("Enter Image Path:")
        print(child.before)
        logging.warning("Write %s", out)
        shutil.move(default_img_with_detection, out)
        print(child.after)
    child.sendeof()
    child.wait()


def main():
    """Prog entry"""
    cfg_file = os.path.join(DRIVENET_DIR, "data/yolo/yolov3.cfg")
    data_file = os.path.join(DARKNET_DIR, "cfg/drivenet_fov60.data")
    weights_file = os.path.join(DRIVENET_DIR, "data/yolo/yolov3_b1.weights")
    darknet_exe = os.path.join(DARKNET_DIR, "build/darknet")

    parser = argparse.ArgumentParser()
    parser.add_argument("--darknet-exe", default=darknet_exe)
    parser.add_argument("--yolo-data-file", default=data_file)
    parser.add_argument("--yolo-cfg-file", default=cfg_file)
    parser.add_argument("--yolo-weights", default=weights_file)
    parser.add_argument("--yolo-result-json", default="/tmp/yolo_result.json")
    parser.add_argument("--image-filenames", default="/tmp/weakness_detection_image_list.txt")
    args = parser.parse_args()
    image_filenames = process_filenames_arg(args.image_filenames)
    detect_by_yolo(args.darknet_exe, args.yolo_data_file, args.yolo_cfg_file,
                   args.yolo_weights, args.yolo_result_json, image_filenames)

if __name__ == "__main__":
    main()
