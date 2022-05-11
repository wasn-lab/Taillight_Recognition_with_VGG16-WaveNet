"""
Train our LSTM on extracted features.
"""
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys
from matplotlib import pyplot
import argparse


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
	tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

	# Helper: Stop when we stop learning.
	early_stopper = EarlyStopping(patience=10)
	# early_stopper = EarlyStopping(patience=15)

	# Helper: Save results.
	timestamp = time.time()
	csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
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
	train_split_percent=0.85
	train_data, test_data = data.split_train_test()
	steps_per_epoch = (len(train_data)* train_split_percent * 0.9) // batch_size
	validation_step = (len(train_data)* (1-train_split_percent) * 0.9) // batch_size
	print(validation_step)

	if load_to_memory:
		# Get data.
		X, y = data.get_all_sequences_in_memory('train', data_type)
		X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
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
		history = rm.model.fit_generator(
			generator=generator,
			steps_per_epoch=steps_per_epoch,
			epochs=nb_epoch,
			verbose=1,
			callbacks=[tb, early_stopper, csv_logger, checkpointer],
			# callbacks=[early_stopper, checkpointer],
			validation_data=val_generator,
			validation_steps=validation_step,
			workers=4)
	# pyplot.plot(history.history['loss'][500:])
	# pyplot.plot(history.history['val_loss'][500:])
	# pyplot.title('model train vs validation loss')
	# pyplot.ylabel('loss')
	# pyplot.xlabel('epoch')
	# pyplot.legend(['train', 'validation'], loc='upper right')
	# pyplot.show()

def main():
	"""These are the main training settings. Set each before running
	this file."""
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--seq_length", type=int, default=20,
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
	saved_model = './data/checkpoints_0428_fine_turn_with_0427/lstm-images.011-0.028.hdf5'
	# saved_model = None  # None or weights file

	load_to_memory = False # pre-load the sequences into memory
	batch_size = 8
	nb_epoch = 1000
	data_type = 'images'
	image_shape = (image_height, image_width, 3)

	# extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

	train(data_type, seq_length, model, saved_model=saved_model,
		class_limit=class_limit, image_shape=image_shape,
		load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
