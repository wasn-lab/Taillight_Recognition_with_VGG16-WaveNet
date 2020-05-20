#!/usr/bin/env python
import os
import argparse
import logging
import tempfile
import tarfile
import pprint
import shutil
import json
import io

import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

from extract_image_labels import LABEL_TYPE_DICT


def untar(tar_filename):
    if not os.path.isfile(tar_filename):
        logging.error("Cannot load tar file: %s", tar_filename)
        return ""
    tempdir = tempfile.mkdtemp()
    tar = tarfile.TarFile(name=tar_filename)
    logging.warning("untar at %s", tempdir)
    tar.extractall(path=tempdir)
    return tempdir


def analyze_tfrecords(tfrecord_filenames):
    docs = []
    for tfrecord in tfrecord_filenames:
        _, basename = os.path.split(tfrecord)
        logging.warning("Process %s", tfrecord)
        doc = {"tfrecord": basename, "camera_object_counts": {}, "laser_object_counts": {}}
        dataset = tf.data.TFRecordDataset(tfrecord, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            stats = frame.context.stats
            if stats.time_of_day:
                doc["time_of_day"] = stats.time_of_day
            if stats.location:
                doc["location"] = stats.location
            if stats.weather:
                doc["weather"] = stats.weather
            for item in frame.context.stats.laser_object_counts:
                type_name = LABEL_TYPE_DICT[item.type]
                doc["laser_object_counts"][type_name] = doc["laser_object_counts"].get(type_name, 0) + item.count
            for item in frame.context.stats.camera_object_counts:
                type_name = LABEL_TYPE_DICT[item.type]
                doc["camera_object_counts"][type_name] = doc["camera_object_counts"].get(type_name, 0) + item.count
        docs.append(doc)
    return docs

def dump_tar_info(tar_filename):
    tempdir = untar(tar_filename)
    tfrecord_filenames = [os.path.join(tempdir, _) for _ in os.listdir(tempdir) if _.endswith(".tfrecord")]
    docs = analyze_tfrecords(tfrecord_filenames)
    pprint.pprint(docs)
    output_file = tar_filename[0:-3] + "json"
    with io.open(output_file, "w", encoding="utf-8") as _fp:
        logging.warning("Write %s", output_file)
        json.dump(docs, _fp, sort_keys=True)
    logging.warning("rm -r %s", tempdir)
    shutil.rmtree(tempdir)


def main():
    tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", help=".tar file of waymo dataset", required=True)
    args = parser.parse_args()
    dump_tar_info(args.tar)


if __name__ == "__main__":
    main()
