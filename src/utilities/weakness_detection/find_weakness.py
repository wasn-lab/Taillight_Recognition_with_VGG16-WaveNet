#!/usr/bin/env python
import argparse
import os
import logging
import shutil
from yolo_mgr import YoloMgr

def move_weakest_images(filenames, weakness_dir):
    """Move weakest images to |weakness_dir|"""
    for filename in filenames:
        deeplab_overlay_fn = filename[:-4] + "_deeplab_overlay.jpg"
        yolo_result_fn = filename[:-4] + "_yolo.jpg"
        for src in [filename, deeplab_overlay_fn, yolo_result_fn]:
            _, basename = os.path.split(src)
            dest = os.path.join(weakness_dir, basename)
            logging.warning("cp %s %s", src, dest)
            shutil.copyfile(src, dest)


def main():
    parser = argparse.ArgumentParser()
    dft_yolo_result = os.path.join(
        os.environ.get("TMP_DIR", "/tmp"),
        "yolo_result.json")
    dft_weakest_image_dir = os.path.join(
        os.environ.get("TMP_DIR", "/tmp"),
        "weakness_images")
    parser.add_argument("--yolo-result-json", default=dft_yolo_result)
    parser.add_argument("--weakness-image-dir", default=dft_weakest_image_dir)
    args = parser.parse_args()

    weakness_dir = args.weakness_image_dir
    if not os.path.isdir(weakness_dir):
        os.makedirs(weakness_dir)

    mgr = YoloMgr(args.yolo_result_json)
    mgr.find_weakness_images()
    mgr.save_weakness_logs(weakness_dir)

    move_weakest_images(mgr.get_weakest_images(), weakness_dir)


if __name__ == "__main__":
    main()
