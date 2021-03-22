import os
import sys
import glob
import csv
from shutil import copyfile
import cv2


def add_frames(train_or_test, classname, path):

	seq_len = 20

	nb_frames = 0

	data_files = []

	for root, dirs, files in os.walk(path):
		#print (root, dirs, files)
		frames = sorted(glob.glob(os.path.join(root, '*png')))
		frames += sorted(glob.glob(os.path.join(root, '*jpg')))
		nb_frames = len(frames)

		if(nb_frames == 0):
			continue

		last_seq_idx = int(get_last_seq_idx(train_or_test, classname))
		print("last_seq_idx: %d" % last_seq_idx)

		

		if(nb_frames <= seq_len):
			#print(nb_frames)
			#print(frames)
			frame_cnt = 0
			filename_no_ext = train_or_test + "_" + classname + "_" + str(last_seq_idx + 1)
			for i in range(seq_len):
				filename = filename_no_ext + "-%04d.jpg" % i
				src = os.path.join(frames[i % nb_frames])
				dst = os.path.join(train_or_test, classname, filename)
				print("[%s] --> [%s]" % (src, dst))
				transfer_file(src, dst, (224, 224)) # resize and copy
				frame_cnt += 1

			data_files.append([train_or_test, classname, filename_no_ext, frame_cnt])

		if(nb_frames > seq_len):
			for i in range((nb_frames//seq_len)*seq_len):
				seq_idx = last_seq_idx + (i // seq_len) + 1;
				#print("i: %d" % i)
				#print("seq_idx: %d" % seq_idx)
				filename_no_ext = train_or_test + "_" + classname + "_" + str(seq_idx)
				filename = filename_no_ext + "-%04d.jpg" % (i%seq_len)
				src = os.path.join(frames[i])
				dst = os.path.join(train_or_test, classname, filename)
				print("[%s] --> [%s]" % (src, dst))
				transfer_file(src, dst, (224, 224)) # resize and copy

				if(len(data_files) != 0):
					data_files_last_seq_idx = int(data_files[-1][2].split('_')[-1])		# ex: get 13 from "train_no_signal_13"
				else:
					data_files_last_seq_idx = 0

				if(len(data_files) == 0 or data_files_last_seq_idx != seq_idx):
					data_files.append([train_or_test, classname, filename_no_ext, seq_len])

	

	print(data_files)


	# update data file
	with open('data_file.csv', 'a') as fout:
		writer = csv.writer(fout)
		writer.writerows(data_files)


def get_last_seq_idx(train_or_test, classname):
	indexes = []
	file_pathes = glob.glob(os.path.join(train_or_test, classname, '*jpg'))

	for f_path in file_pathes:
		f = f_path.split(os.path.sep)
		f_name = f[2]
		f_name_seq = f_name.split('-')[0]
		f_name_idx = int(f_name_seq.split('_')[-1])
		if f_name_idx not in indexes:
			indexes.append(f_name_idx)

	if not indexes:
		return -1
	else:
		return(max(indexes))

def transfer_file(src, dst, dim=(224, 224)):
	img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
	#print(type(img))
	#print('Original Dimensions : ',img.shape)
	#width = 224
	#height = 224
	#dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#print('Resized Dimensions : ',resized.shape)
	ret = cv2.imwrite(dst, resized)
	if ret is not True:
		print("!!! [%s] fail to write !!!" % dst)

def main():
	if (len(sys.argv) == 4):
		train_or_test = sys.argv[1]
		classname = sys.argv[2]
		extract_path = sys.argv[3]

		seq_path = os.path.join(train_or_test, classname)
		if os.path.exists(seq_path) is not True:
			os.makedirs(seq_path)

		add_frames(train_or_test, classname, extract_path)
	else:
		print ("invalid argument")
		print ("Usage: python traverse.py [train_or_test] [class_name] [path]")

if __name__ == '__main__':
    main()

