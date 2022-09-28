"""
Process an image that we can pass to our networks.
"""
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
# from tensorflow.python.keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from matplotlib import pyplot as plt


from skimage import color
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import cv2 as cv

# np.set_printoptions(threshold=10)

def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    """
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)

    # img_arr = np.expand_dims(img_arr, axis=0)
    x = preprocess_input(img_arr)
    # x = (img_arr / 255.).astype(np.float32)
    """
    # print(image)
    # fig=plt.figure()
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))
    img_arr = img_to_array(image)
    # img_arr = img_as_float(image)
    # fig.add_subplot(2, 1, 1)
    # plt.imshow(img_arr)
    # segments = slic(img_arr, n_segments = 500, sigma = 0.5)
    # superpixels = color.label2rgb(segments, img_arr, kind='avg')
    # fig.add_subplot(2, 1, 2)
    # plt.imshow(superpixels)
    # plt.show()
    x = preprocess_input(img_arr)
    # x = (img_arr / 255.).astype(np.float32)
    # x = img_arr 
    # print(x)
    # x = (img_arr-np.amin(img_arr))/(np.amax(img_arr)-np.amin(img_arr))
    # plt.imshow(x)
    # plt.show()
    return x
