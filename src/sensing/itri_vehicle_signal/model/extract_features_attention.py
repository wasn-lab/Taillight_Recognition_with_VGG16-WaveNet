"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
from extractor_attention import Extractor
from attention_model import Attention_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from tqdm import tqdm

def extract_features(seq_length=40, class_limit=2, image_shape=(224, 224, 3)):

	# Get the dataset.
	data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

	# get the model.
	model = Extractor(image_shape=image_shape)
	attn_model = Attention_model(image_shape=image_shape)
	# Loop through data.
	pbar = tqdm(total=len(data.data))

	#print(data.data[1000:1010])

	for video in data.data:               
        
		# Get the path to the sequence for this video.
		path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + \
			'-features')  # numpy will auto-append .npy

		# Check if we already have it.
		if os.path.isfile(path + '.npy'):
			pbar.update(1)
			continue

		# Get the frames for this video.
		frames = data.get_frames_for_sample(video)
        
		#print(frames)
		# Now downsample to just the ones we need.
		frames = data.rescale_list(frames, seq_length)

		# Now loop through and extract features to build the sequence.
		sequence = []
		for image_path in frames:
			img = image.load_img(image_path)
			features = attn_model.attention(img)
			img_array = image.img_to_array(img)
			features = features * img_array
			features = model.extract_image(features)
			sequence.append(features)

		# Save the sequence.
		np.save(path, sequence)

		pbar.set_description("Processing %s" % image_path.split('/')[-1])
		pbar.update(1)

	pbar.close()
