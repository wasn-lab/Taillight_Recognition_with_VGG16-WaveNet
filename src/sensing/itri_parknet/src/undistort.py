import os
import argparse
import logging
import cv2
import numpy as np

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1208

def _init_params():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    parknet_dir = os.path.join(cur_dir, "..")
    camera_utils_dir = os.path.join(parknet_dir, "..", "camera_utils")
    ymlfile = os.path.join(camera_utils_dir, "data", "sf3324.yml")
    fs = cv2.FileStorage(ymlfile, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    distortion_coefficients = fs.getNode("distortion_coefficients").mat()
    return camera_matrix, distortion_coefficients


def _init_mapx_mapy(ret=list()):
    if ret:
        return ret[0], ret[1]
    camera_matrix, distortion_coefficients = _init_params()
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, camera_matrix, (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    ret = [mapx, mapy]
    return ret[0], ret[1]

def _undistort_by_dir(dirname):
    for root, _dir, files in os.walk(dirname):
        for fn in files:
            if fn.endswith(".jpg") and "undistort_" not in fn:
                fullpath = os.path.join(fn)
                _undistort_by_filename(fullpath)


def _undistort_by_filename(filename):
    mapx, mapy = _init_mapx_mapy()
    img = cv2.imread(filename)
    undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    path, basename = os.path.split(filename)
    new_filename = "undistort_{}".format(basename)
    fullpath = os.path.join(path, new_filename)
    logging.warning("Write %s", fullpath)
    cv2.imwrite(fullpath, undistorted)


def _redistort_by_filename(filename):
    camera_matrix, distortion_coefficients = _init_params()
    img = cv2.imread(filename)
    rvec = np.array([0, 0, 0])
    tvec = np.array([0, 0, 0])
    result = np.zeros_like(img)
    cv2.projectPoints(img, rvec, tvec, camera_matrix, distortion_coefficients, result);

    path, basename = os.path.split(filename)
    new_filename = "redistort_{}".format(basename)
    fullpath = os.path.join(path, new_filename)
    logging.warning("Write %s", fullpath)
    cv2.imwrite(fullpath, result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f")
    parser.add_argument("--dir", "-d")
    args = parser.parse_args()

    if args.dir:
        _undistort_by_dir(args.dir)
    if args.filename:
        _undistort_by_filename(args.filename)

if __name__ == "__main__":
    main()
