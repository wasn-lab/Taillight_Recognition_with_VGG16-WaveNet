#!/usr/bin/env python
import argparse
from yolo_mgr import YoloMgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-result-json", default="/tmp/yolo_result.json")
    args = parser.parse_args()

    mgr = YoloMgr(args.yolo_result_json)
    mgr.find_weakness_images()


if __name__ == "__main__":
    main()
