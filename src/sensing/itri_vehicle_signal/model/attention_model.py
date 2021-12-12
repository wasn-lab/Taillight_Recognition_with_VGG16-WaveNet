import sys
import numpy as np
import os.path
from data import DataSet
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from keras.preprocessing.image import img_to_array, load_img, array_to_img

from keras import models
from keras import layers
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation

class Attention_model():
	def __init__(self, image_shape=(224, 224, 3), weights=None):
		"""Either load pretrained from imagenet, or load our saved
		weights from our own training."""

		self.weights = weights  # so we can check elsewhere which model

		input_tensor = Input(image_shape)
		# Get model with pretrained weights.
		base_model = Conv2D(32, (3, 3), dilation_rate=1, strides=(1, 1), padding="same")(input_tensor)
		base_model = Conv2D(64, (3, 3), dilation_rate=2, strides=(1, 1), padding="same")(base_model)
		base_model = Conv2D(64, (3, 3), dilation_rate=2, strides=(1, 1), padding="same")(base_model)
		base_model = Conv2D(3, (3, 3), dilation_rate=1, strides=(1, 1), padding="same")(base_model)

		# base_model.trainable = True	# todo: find-tune the model

		# We'll extract features at the final pool layer.
		self.model = Model(
			inputs=input_tensor,
			outputs=base_model
		)
		# summarize layers
		print(self.model.summary())

	def attention(self, img):
		return self.extract_image(img)

	def extract_image(self, img):
		if not isinstance(img, np.ndarray):
			x = image.img_to_array(img)
		else :
			x = img
		x = np.expand_dims(x, axis=0)
		#x = preprocess_input(x)

		# Get the prediction.
		features = self.model.predict(x)

		if self.weights is None:
			# For imagenet/default network:
			features = features[0]
		else:
			# For loaded network:
			features = features[0]

		features = (features-np.amin(features))/(np.amax(features)-np.amin(features))*5

		return features
