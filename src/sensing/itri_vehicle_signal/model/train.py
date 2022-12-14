"""
Train our LSTM on extracted features.
"""
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend
import tensorflow as keras
from models import ResearchModels
from data import DataSet
import time
import os.path
import sys
from matplotlib import pyplot as plt
import argparse
import os
import datetime

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)


# class LRTensorBoard(TensorBoard):
#     # add other arguments to __init__ if you need
#     def __init__(self, log_dir, **kwargs):
#         super().__init__(log_dir=log_dir, **kwargs)

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs.update({'lr': backend.eval(self.model.optimizer.lr)})
#         super().on_epoch_end(epoch, logs)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print("current decayed lr: {:0.13f}".format(current_decayed_lr))

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):

	# Helper: Save the model.
	checkpointer = ModelCheckpoint(
		filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
			'.{epoch:03d}-{val_loss:.3f}.hdf5'),
			verbose=1,
			save_best_only=True)

	# Helper: TensorBoard
	log_dir = "data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tb = TensorBoard(log_dir=log_dir)
	# tb = TensorBoard(log_dir=os.path.join('data', 'logs'))

	# Helper: CustomCallback - learning rate
	curr_lr = CustomCallback()

	# Helper: Stop when we stop learning.
	early_stopper = EarlyStopping(monitor='val_loss', patience=5)
	# early_stopper = EarlyStopping(patience=15)

	# Helper: Save results.
	timestamp = time.time()
	csv_logger = CSVLogger(os.path.join('data', 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), model + '-' + 'training-' + \
	   str(timestamp) + '.log'))

	# Get the data and process it.
	if image_shape is None:
		data = DataSet(
			seq_length=seq_length,
			class_limit=class_limit
			)
	else:
		data = DataSet(
			seq_length=seq_length,
			class_limit=class_limit,
			image_shape=image_shape
			)

	# Get samples per epoch.
	# Multiply by 0.7 to attempt to guess how much of data.data is the train set.
	train_split_percent=0.7
	train_data, test_data = data.split_train_test()
	steps_per_epoch = (len(train_data) * 0.25) // batch_size
	validation_step = (len(test_data) * 0.5) // batch_size
	steps_per_epoch = 7356//batch_size
	validation_step = 176//batch_size
	print(validation_step)
	# input("show val step")

	if load_to_memory:
		# Get data.
		X, y = data.get_all_sequences_in_memory('train', data_type, train_split_percent)
		X_test, y_test = data.get_all_sequences_in_memory('test', data_type, train_split_percent)
	else:
		# Get generators.
		generator = data.frame_generator(batch_size, 'train', data_type, train_split_percent)
		val_generator = data.frame_generator(batch_size, 'test', data_type, train_split_percent)

	# Get the model.
	rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

	# Fit!
	if load_to_memory:
		# Use standard fit.
		rm.model.fit(
			X,
			y,
			batch_size=batch_size,
			validation_data=(X_test, y_test),
			verbose=1,
			callbacks=[tb, early_stopper, csv_logger, checkpointer],
			# callbacks=[early_stopper, checkpointer],
			epochs=nb_epoch)
	else:
		# Use fit generator.
		H = rm.model.fit_generator(
			generator=generator,
			steps_per_epoch=steps_per_epoch,
			epochs=nb_epoch,
			verbose=1,
			callbacks=[tb, early_stopper, csv_logger, checkpointer, curr_lr],
			# callbacks=[early_stopper, checkpointer],
			validation_data=val_generator,
			validation_steps=validation_step,
			workers=4)

	plt.style.use("ggplot")
	plt.figure()
	plt.plot( H.history["loss"], label="train_loss")
	plt.plot( H.history["val_loss"], label="val_loss")
	plt.plot( H.history["categorical_accuracy"], label="train_acc")
	plt.plot( H.history["val_categorical_accuracy"], label="val_acc")
	# plt.plot( H.history["turnlight_output_categorical_accuracy"], label="train_acc")
	# plt.plot( H.history["val_turnlight_output_categorical_accuracy"], label="val_acc")
	# plt.plot( H.history["val_Lturnlight_output_binary_accuracy"], label="val_L_acc")
	# plt.plot( H.history["val_Rturnlight_output_binary_accuracy"], label="val_R_acc")
	plt.title("Training Loss and Accuracy on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.show()


def main():
	"""These are the main training settings. Set each before running
	this file."""
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--seq_length", type=int, default=32,
						help="the length of a sequence")
	parser.add_argument("--class_limit", type=int, default=4,
						help="how much classes need to clasify")
	parser.add_argument("--image_height", type=int, default=224,
						help="how to reszie the image height")
	parser.add_argument("--image_width", type=int, default=224,
						help="how to reszie the image width")
	args = parser.parse_args()

	seq_length = args.seq_length
	class_limit = args.class_limit
	image_height = args.image_height
	image_width = args.image_width


	"""
	sequences_dir = os.path.join('data', 'sequences')
	if not os.path.exists(sequences_dir):
		os.mkdir(sequences_dir)
	"""
	checkpoints_dir = os.path.join('data', 'checkpoints')
	if not os.path.exists(checkpoints_dir):
		os.mkdir(checkpoints_dir)


	# model can be only 'lstm'
	model = 'lstm'
	saved_model = './data/lstm-images_9e508_9e308.hdf5'
	saved_model = None  # None or weights file

	load_to_memory = False # pre-load the sequences into memory
	batch_size = 2
	nb_epoch = 1000
	data_type = 'images'
	image_shape = (image_height, image_width, 3)

	train(data_type, seq_length, model, saved_model=saved_model,
		class_limit=class_limit, image_shape=image_shape,
		load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
