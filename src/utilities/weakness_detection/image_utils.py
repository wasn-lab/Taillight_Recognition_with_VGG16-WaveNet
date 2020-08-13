"""Utilities functions about image"""
import cv2

def get_image_size(filename, cache=dict()):
    """Return (width, height) """
    if filename in cache:
        return cache[filename]
    img = cv2.imread(filename)
    cache[filename] = (img.shape[1], img.shape[0])
    return cache[filename]
