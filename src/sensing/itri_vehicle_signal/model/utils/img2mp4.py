import os
import sys
import cv2
import numpy as np
import glob


width = 224
height = 224


def img2mp4(path, output_file):


	for root, dirs, files in os.walk(path):
		img_files = sorted(glob.glob(os.path.join(root, '*png')))
		img_files += sorted(glob.glob(os.path.join(root, '*jpg')))

	# img = cv2.imread(img_files[0])
	# height, width, layers = img.shape
	print("(width, height) = (%d, %d)" % (width, height))
	size = (width, height)

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	video_writer = cv2.VideoWriter(output_file, fourcc, 30, size)

	for file in img_files:
		img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
		resized_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
		video_writer.write(resized_img)

	video_writer.release()


def main():
	if (len(sys.argv) == 3):
		path = sys.argv[1]
		output_file = sys.argv[2]
		img2mp4(path, output_file)
	else:
		print ("invalid argument")
		print ("Usage: python img2mp4.py [path] [output_file]")

if __name__ == '__main__':
    main()
