"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    # img_arr = np.expand_dims(img_arr, axis=0)
    x = preprocess_input(img_arr)
    # x = (img_arr / 255.).astype(np.float32)

    return x
