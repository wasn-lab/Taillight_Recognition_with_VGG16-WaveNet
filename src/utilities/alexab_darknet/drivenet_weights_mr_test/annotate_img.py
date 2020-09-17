#!/usr/bin/env python
import argparse
import json
import io
import logging
import os


def _read_existing_data(filename):
    jdata = []
    if os.path.isfile(filename):
        with io.open(filename, encoding="utf-8") as _fp:
            contents = _fp.read()
            jdata = json.loads(contents)
    return jdata


def _write_annotation(filename, jdata):
    logging.warn("Write %s", filename)
    with io.open(filename, "w", encoding="utf-8") as _fp:
        _fp.write(json.dumps(jdata, sort_keys=True))


def _gen_annotation_entry(args):
    doc = {"negative": args.negative,
           "class_id": args.class_id,
           "iou": args.iou,
           "coordinates": args.coordinates}
    return doc


def annotate_img(args):
    filename = args.filename
    if not filename.endswith(".json"):
        filename = filename[:-4] + ".json"
    jdata = _read_existing_data(filename)
    doc = _gen_annotation_entry(args)
    jdata.append(doc)
    _write_annotation(filename, jdata)


def main():
    # Usage:
    #  python3 annotate_img.py -f 1596507098735017792.json \
    #        --must-have 1 --iou 0.9 --coordinates 437 232 465 332
    # where bbox is left-top coordinate and bottom-right coordinate.
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--class-id", "-c", type=int, default=0,
        help="0:person, 1:bicycle, 2:car, 3:motorbike, 5:bus, 7:truck")
    parser.add_argument("--negative", "-n", action="store_true")
    parser.add_argument("--coordinates", type=int, nargs="+", required=True)
    args = parser.parse_args()
    annotate_img(args)


if __name__ == "__main__":
    main()
